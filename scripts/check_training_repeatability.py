"""Repeat one A CNN refit in isolation; keep original evaluation unchanged."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from csi_pipeline import find_matching_files, make_split, prepare_scenario_data
from evaluate_optuna import load_best_result
from tune_optuna import build_tuning_model, tf


def main(args):
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('This isolated comparison requires the GPU environment.')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    files = {person: find_matching_files(person) for person in 'ABCD'}
    split = make_split(files, 'A', 15)
    best, epochs = load_best_result(ROOT / 'optuna_results_15_20261007/A_cnn_best.json', 'A', 'cnn', split)
    data = prepare_scenario_data(split, window_sec=best['params']['window_sec'])
    output = ROOT / 'data_audit/execution_integrity_15_20261008'
    output.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=data['y_train'])
    tag = 'gpu_deterministic' if args.deterministic else 'gpu_standard'
    records, previous = [], None
    for run in range(args.runs):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_tuning_model(data['X_train'].shape[1:], 'cnn', best['params'])
        initial_hash = hashlib.sha256(b''.join(value.tobytes() for value in model.get_weights())).hexdigest()
        print(f'{tag} run {run+1}/{args.runs}, seed42, initial weights {initial_hash}', flush=True)
        history = model.fit(data['X_train'], data['y_train'], epochs=epochs,
                            batch_size=best['params']['batch_size'], shuffle=True,
                            class_weight=dict(enumerate(weights)), verbose=2)
        train = model.predict(data['X_train'], batch_size=64, verbose=0).argmax(axis=1)
        probabilities = model.predict(data['X_test'], batch_size=64, verbose=0)
        predictions = probabilities.argmax(axis=1)
        row = {'run': run+1, 'seed': 42, 'deterministic': args.deterministic,
               'initial_weights_sha256': initial_hash,
               'train_empty_recall': float(np.mean(train[data['y_train'] == 2] == 2)),
               'test_empty_recall': float(np.mean(predictions[data['y_test'] == 2] == 2)),
               'test_macro_f1': float(f1_score(data['y_test'], predictions, labels=[0, 1, 2], average='macro')),
               'test_matrix': confusion_matrix(data['y_test'], predictions, labels=[0, 1, 2]).tolist()}
        if previous is not None:
            row['changed_test_labels_vs_previous'] = int(np.count_nonzero(predictions != previous.argmax(axis=1)))
            row['max_probability_difference_vs_previous'] = float(np.max(np.abs(probabilities - previous)))
        previous = probabilities.copy()
        pd.DataFrame(history.history).to_csv(output / f'{tag}_{run+1}_history.csv', index=False)
        np.save(output / f'{tag}_{run+1}_test_probabilities.npy', probabilities)
        records.append(row)
        print('Result:', json.dumps(row), flush=True)
        del model
        tf.keras.backend.clear_session()
    (output / f'{tag}_repeats.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--deterministic', action='store_true')
    main(parser.parse_args())
