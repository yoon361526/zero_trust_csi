"""Audit saved Optuna evaluation and current signals; no model fitting."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import (
    CLASS_NAMES, PREPROCESSING_VERSION, find_matching_files, make_split,
    prepare_scenario_data, read_csi_session, validate_unique_recordings,
)
from evaluate_optuna import load_best_result


def main():
    best_dir = ROOT / 'optuna_results_20261007_new'
    evaluation_dir = ROOT / 'optuna_evaluation_20261007_new'
    output = ROOT / 'data_audit/optuna_empty_diagnosis_20261007'
    output.mkdir(exist_ok=True)
    files = {person: find_matching_files(person) for person in 'ABCD'}
    validate_unique_recordings(files)
    summaries, session_rows, signal_rows = [], [], []
    profiles = {}
    for target in 'ABC':
        split = make_split(files, target)
        training = [path for key, paths in split.items() if key.startswith('train_') for path in paths]
        for model_name in ('cnn', 'lstm'):
            tag = f'{target}_{model_name}'
            best, epochs = load_best_result(best_dir / f'{tag}_best.json', target, model_name, split)
            data = prepare_scenario_data(split, window_sec=best['params']['window_sec'])
            saved = pd.read_csv(evaluation_dir / f'{tag}_test_predictions.csv')
            metadata = pd.DataFrame(data['test_metadata'])
            assert len(saved) == len(metadata), tag
            for key in ('Person', 'Session', 'Class', 'Activity'):
                assert saved[key].tolist() == metadata[key].tolist(), (tag, key)
            for key in ('Start_Sec', 'End_Sec'):
                assert np.allclose(saved[key], metadata[key], rtol=0, atol=1e-6), (tag, key)
            norm = np.load(evaluation_dir / f'{tag}_preprocessing.npz')
            assert int(norm['preprocessing_version']) == PREPROCESSING_VERSION
            assert np.allclose(norm['mean'], data['mean'], rtol=1e-6, atol=1e-6), tag
            assert np.allclose(norm['std'], data['std'], rtol=1e-6, atol=1e-6), tag
            true = np.array([CLASS_NAMES.index(value) for value in saved['Class']])
            predicted = np.array([CLASS_NAMES.index(value) for value in saved['Predicted_Class']])
            cm = confusion_matrix(true, predicted, labels=[0, 1, 2])
            stored_cm = pd.read_csv(evaluation_dir / f'cm_{tag}_final_test_counts.csv', index_col=0)
            assert np.array_equal(cm, stored_cm.to_numpy()), tag
            empty = saved[saved['Person'] == 'D']
            summaries.append({'run': tag, 'validation_macro_f1': best['validation_macro_f1'],
                              'refit_epochs': epochs, 'window_sec': best['params']['window_sec'],
                              'empty_recall': float((empty['Predicted_Class'] == 'Empty').mean()),
                              'protocol_matches_current_training_data': True,
                              'saved_normalization_matches': True, 'labels_and_times_match': True,
                              'matrix_counts_match': True})
            for name, part in empty.groupby('Session'):
                counts = part['Predicted_Class'].value_counts()
                session_rows.append({'run': tag, 'file': name, 'windows': len(part),
                                     **{label: int(counts.get(label, 0)) for label in CLASS_NAMES},
                                     'empty_recall': float((part['Predicted_Class'] == 'Empty').mean())})
            for path in training + split['test_empty']:
                if path not in profiles:
                    profiles[path] = np.median(read_csi_session(path).amplitude[:, 2:], axis=0)
            train_profiles = np.stack([profiles[path] for path in training])
            scale = data['std'][2:]
            for path in split['test_empty']:
                distances = np.sqrt(np.mean(((train_profiles - profiles[path]) / scale) ** 2, axis=1))
                row = {'run': tag, 'file': Path(path).name}
                for person in 'ABCD':
                    candidates = [index for index, ref in enumerate(training) if Path(ref).parent.name == person]
                    closest = min(candidates, key=lambda index: distances[index])
                    row[f'nearest_{person}'] = f'{person}/{Path(training[closest]).name}'
                    row[f'distance_{person}'] = float(distances[closest])
                signal_rows.append(row)
            print(f'{tag}: verified current protocol, normalization, labels, times and matrix; '
                  f'validation F1={best["validation_macro_f1"]:.4f}, '
                  f'empty recall={summaries[-1]["empty_recall"]:.4f}, epochs={epochs}', flush=True)
            del data
    pd.DataFrame(session_rows).to_csv(output / 'empty_predictions_by_file.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(signal_rows).to_csv(output / 'empty_signal_distances.csv', index=False, encoding='utf-8-sig')
    report = {'model_training_performed': False, 'runs': summaries,
              'session_predictions': session_rows, 'signal_comparisons': signal_rows,
              'limitation': 'Signal profile distances describe differences, not label identity or the trained model decision. Final model weights and training predictions were not saved, so training accuracy and controlled normalization effects cannot be checked without a separate training experiment.'}
    (output / 'results_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Actual window count varies with the selected 1/2/4-second setting.
    d_profiles = np.stack([profiles[path] for path in files['D']])
    distances = np.sqrt(np.mean((d_profiles[:, None, :] - d_profiles[None, :, :]) ** 2, axis=2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    names = [f'D{i:02d} {"train" if i <= 16 else "test"}' for i in range(1, 21)]
    first = axes[0].imshow(d_profiles, aspect='auto', cmap='viridis')
    axes[0].set(title='Current D recordings: median amplitude profile', xlabel='Amplitude feature index (first two excluded)',
                yticks=np.arange(20), yticklabels=names)
    axes[0].axhline(15.5, color='red')
    fig.colorbar(first, ax=axes[0], label='CSI amplitude')
    second = axes[1].imshow(distances, cmap='magma')
    axes[1].set(title='Between-recording RMS profile difference', xticks=np.arange(20),
                xticklabels=[f'{i:02d}' for i in range(1, 21)],
                yticks=np.arange(20), yticklabels=[f'D{i:02d}' for i in range(1, 21)])
    axes[1].axhline(15.5, color='cyan')
    axes[1].axvline(15.5, color='cyan')
    fig.colorbar(second, ax=axes[1], label='RMS amplitude difference')
    fig.savefig(output / 'current_D_profiles.png', dpi=150)
    plt.close(fig)
    print(pd.DataFrame(session_rows).to_string(index=False))
    print(pd.DataFrame(signal_rows).to_string(index=False))


if __name__ == '__main__':
    main()
