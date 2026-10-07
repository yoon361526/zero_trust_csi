"""Check the actual A CNN inputs and replay inference/refitting on CPU."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from csi_pipeline import find_matching_files, make_split, prepare_scenario_data, read_csi_session
from evaluate_optuna import load_best_result
from tune_optuna import build_tuning_model, tf


def independent_decode(path):
    """Use CSV fields rather than the pipeline's payload/header regex."""
    origin, timestamps, payloads = None, [], []
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line in stream:
            pieces = line.rstrip('\r\n').split('\t', 2)
            if len(pieces) != 3:
                continue
            try:
                timestamp = int(pieces[0])
            except ValueError:
                continue
            origin = timestamp if origin is None else min(origin, timestamp)
            fields = next(csv.reader([pieces[2]]))
            if len(fields) != 25 or fields[0] != 'CSI_DATA':
                continue
            try:
                length, invalid = int(fields[22]), int(fields[23])
                payload = np.fromstring(fields[24].strip('[]'), sep=',', dtype=np.float32)
            except ValueError:
                continue
            if length != 384 or len(payload) != 384 or invalid not in (0, 1):
                continue
            if invalid:
                payload[:4] = 0
            timestamps.append(timestamp)
            payloads.append(payload)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    payloads = np.asarray(payloads, dtype=np.float32)
    order = np.argsort(timestamps, kind='stable')
    timestamps, indices = np.unique(timestamps[order], return_index=True)
    payloads = payloads[order][indices].astype(np.float64)
    amplitudes = np.sqrt(payloads[:, 0::2] ** 2 + payloads[:, 1::2] ** 2).astype(np.float32)
    return (timestamps - origin).astype(np.float64) / 1e9, amplitudes


def independent_empty_windows(times, amplitude, window_sec, fps=20):
    """All D files cover a single 180-second empty-room segment."""
    keep = (times >= 0) & (times < 180)
    times, amplitude = times[keep], amplitude[keep]
    first = int(np.ceil(times[0] * fps - 1e-9))
    last = int(np.floor(times[-1] * fps + 1e-9))
    grid = np.arange(first, last + 1, dtype=np.float64) / fps
    sampled = np.stack([np.interp(grid, times, amplitude[:, k])
                        for k in range(amplitude.shape[1])], axis=1).astype(np.float32)
    right = np.minimum(np.searchsorted(times, grid), len(times) - 1)
    left = np.maximum(right - 1, 0)
    supported = np.isclose(grid, times[right], rtol=0, atol=1e-9) | (times[right] - times[left] <= .5 + 1e-9)
    width = int(window_sec * fps)
    return np.asarray([sampled[i:i + width] for i in range(0, len(sampled) - width + 1, fps)
                       if supported[i:i + width].all()], dtype=np.float32)


def main(args):
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    assert not tf.config.list_physical_devices('GPU')
    output = ROOT / 'data_audit/execution_integrity_15_20261008'
    output.mkdir(parents=True, exist_ok=True)
    evaluation = ROOT / 'optuna_evaluation_15_20261007'
    files = {person: find_matching_files(person) for person in 'ABCD'}
    split = make_split(files, 'A', 15)
    best, epochs = load_best_result(ROOT / 'optuna_results_15_20261007/A_cnn_best.json', 'A', 'cnn', split)
    data = prepare_scenario_data(split, window_sec=best['params']['window_sec'])
    report = {'execution_device': 'CPU', 'gpu_devices': [], 'epochs': epochs, 'parameters': best['params']}
    for partition in ('train', 'test'):
        expected_labels = np.array([2 if Path(row['Source_File']).parent.name == 'D'
                                   else 0 if Path(row['Source_File']).parent.name == 'A' else 1
                                   for row in data[f'{partition}_metadata']])
        np.testing.assert_array_equal(expected_labels, data[f'y_{partition}'])
    report['folder_to_numeric_label_mapping_matches'] = True
    with np.load(evaluation / 'A_cnn_preprocessing.npz') as stored:
        np.testing.assert_allclose(stored['mean'], data['mean'], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(stored['std'], data['std'], atol=1e-6, rtol=1e-6)
    manifest = json.loads((evaluation / 'evaluation_manifest.json').read_text())
    a_job = next(job for job in manifest if job['authorized_person'] == 'A' and job['model'] == 'cnn')
    for record in a_job['train_recordings'] + a_job['test_recordings']:
        with open(record['file'], 'rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    report['input_hashes_and_training_normalization_match'] = True
    decoded, independent, parser_checks = {}, [], []
    for path in split['train_empty'] + split['test_empty']:
        times, amplitude = independent_decode(path)
        original = read_csi_session(path)
        np.testing.assert_array_equal(times, original.elapsed_sec)
        np.testing.assert_allclose(amplitude, original.amplitude, atol=2e-5, rtol=1e-6)
        windows = independent_empty_windows(times, amplitude, best['params']['window_sec'])
        decoded[path] = (windows - data['mean']) / data['std']
        parser_checks.append({'session': Path(path).name, 'packets': len(times),
                             'max_amplitude_difference': float(np.max(np.abs(amplitude - original.amplitude)))})
        if path in split['test_empty']:
            independent.append(decoded[path])
    report['independent_csv_parser_comparison'] = parser_checks
    independent = np.concatenate(independent)
    empty_mask = data['y_test'] == 2
    np.testing.assert_allclose(independent, data['X_test'][empty_mask], atol=3e-5, rtol=1e-5)
    report['independent_test_D_windows_match'] = True
    print('Independent raw CSV decoding, D windows, labels, hashes and normalization: PASS', flush=True)
    model = tf.keras.models.load_model(evaluation / 'A_cnn_model.keras', compile=False)
    saved = pd.read_csv(evaluation / 'A_cnn_test_predictions.csv')
    saved_probabilities = saved[['Probability_Auth', 'Probability_Unauth', 'Probability_Empty']].to_numpy()
    cpu_probabilities = model.predict(data['X_test'], batch_size=64, verbose=0)
    cpu_predictions = cpu_probabilities.argmax(axis=1)
    saved_predictions = saved_probabilities.argmax(axis=1)
    report['cpu_replay_max_probability_difference'] = float(np.max(np.abs(cpu_probabilities - saved_probabilities)))
    report['cpu_replay_changed_labels'] = int(np.count_nonzero(cpu_predictions != saved_predictions))
    report['cpu_replay_test_matrix'] = confusion_matrix(data['y_test'], cpu_predictions, labels=[0, 1, 2]).tolist()
    different_batch = model.predict(independent, batch_size=1, verbose=0)
    direct = model(independent[:8], training=False).numpy()
    report['independent_D_cpu_batch1_prediction_counts'] = np.bincount(different_batch.argmax(axis=1), minlength=3).tolist()
    report['batch1_vs_batch64_D_changed_labels'] = int(np.count_nonzero(different_batch.argmax(axis=1) != cpu_predictions[empty_mask]))
    report['direct_call_vs_predict_max_probability_difference'] = float(np.max(np.abs(direct - different_batch[:8])))
    print('CPU replay:', json.dumps({k: v for k, v in report.items() if k.startswith('cpu_') or 'batch' in k}), flush=True)
    del model
    tf.keras.backend.clear_session()
    if args.cpu_refit:
        tf.keras.utils.set_random_seed(42)
        fresh = build_tuning_model(data['X_train'].shape[1:], 'cnn', best['params'])
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=data['y_train'])
        print(f'Fresh CPU refit: {epochs} epochs, fixed selected parameters; no test in fit', flush=True)
        history = fresh.fit(data['X_train'], data['y_train'], epochs=epochs,
                            batch_size=best['params']['batch_size'], shuffle=True,
                            class_weight=dict(enumerate(weights)), verbose=2)
        pd.DataFrame(history.history).to_csv(output / 'cpu_refit_history.csv', index=False)
        train_probabilities = fresh.predict(data['X_train'], batch_size=64, verbose=0)
        test_probabilities = fresh.predict(data['X_test'], batch_size=64, verbose=0)
        train_pred, test_pred = train_probabilities.argmax(axis=1), test_probabilities.argmax(axis=1)
        report['fresh_cpu_refit'] = {
            'train_empty_recall': float(np.mean(train_pred[data['y_train'] == 2] == 2)),
            'test_empty_recall': float(np.mean(test_pred[empty_mask] == 2)),
            'test_macro_f1': float(f1_score(data['y_test'], test_pred, labels=[0, 1, 2], average='macro')),
            'test_matrix': confusion_matrix(data['y_test'], test_pred, labels=[0, 1, 2]).tolist(),
            'finite_history_and_predictions': bool(all(np.isfinite(v).all() for v in history.history.values())
                                                   and np.isfinite(test_probabilities).all()),
        }
        fresh.save(output / 'fresh_cpu_A_cnn.keras')
        pd.DataFrame(test_probabilities, columns=['Auth', 'Unauth', 'Empty']).to_csv(output / 'cpu_refit_test_probabilities.csv', index=False)
        print('Fresh CPU result:', json.dumps(report['fresh_cpu_refit']), flush=True)
    (output / 'execution_integrity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-refit', action='store_true')
    main(parser.parse_args())
