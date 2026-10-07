"""Rebuild preprocessing and independently verify saved labels and matrices."""

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import find_matching_files, make_split, prepare_scenario_data, validate_unique_recordings

LABELS = ['Auth', 'Unauth', 'Empty']


def main():
    predictions = pd.read_csv(ROOT / 'csi_test_window_predictions.csv')
    saved_splits = pd.read_csv(ROOT / 'csi_session_split.csv')
    files = {person: find_matching_files(person) for person in 'ABCD'}
    validate_unique_recordings(files)
    expected_labels = np.where(predictions['Person'] == 'D', 'Empty',
                               np.where(predictions['Person'] == predictions['Scenario'].str[0], 'Auth', 'Unauth'))
    report = {'checked_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
              'model_training_performed': False,
              'preprocessor_source_sha256': hashlib.sha256((ROOT / 'csi_pipeline.py').read_bytes()).hexdigest(),
              'prediction_rows': len(predictions),
              'label_mismatches': int(np.count_nonzero(predictions['Class'] != expected_labels)),
              'scenarios': {}, 'matrix_checks': []}
    assert report['label_mismatches'] == 0, 'Saved labels differ from folder roles'
    totals = {model: np.zeros((3, 3), dtype=int) for model in ['1D-CNN', 'LSTM']}
    for target in 'ABC':
        split = make_split(files, target)
        data = prepare_scenario_data(split)
        train = {Path(path).name for name, paths in split.items() if name.startswith('train_') for path in paths}
        test = {Path(path).name for name, paths in split.items() if name.startswith('test_') for path in paths}
        assert not train & test
        reserved = saved_splits[(saved_splits['Scenario'] == f'{target} Auth') &
                               (saved_splits['Partition'] == 'test')]
        assert set(reserved['Session']) == test
        old_stats = np.load(ROOT / f'csi_preprocessing_{target}.npz')
        same_mean = bool(np.allclose(old_stats['mean'], data['mean'], atol=1e-6))
        same_std = bool(np.allclose(old_stats['std'], data['std'], atol=1e-6))
        metadata = pd.DataFrame(data['test_metadata'])
        checked = {'train_files': len(train), 'test_files': len(test),
                   'train_windows': len(data['X_train']), 'test_windows': len(data['X_test']),
                   'labels_match_metadata': all(LABELS[int(label)] == row['Class']
                                                for label, row in zip(data['y_test'], data['test_metadata'])),
                   'normalization_matches_saved_run': same_mean and same_std,
                   'finite_train_and_test': bool(np.isfinite(data['X_train']).all() and np.isfinite(data['X_test']).all())}
        for model, tag in [('1D-CNN', '1d_cnn'), ('LSTM', 'lstm')]:
            subset = predictions[(predictions['Scenario'] == f'{target} Auth') & (predictions['Model'] == model)].reset_index(drop=True)
            assert len(subset) == len(metadata)
            for column in ['Person', 'Session', 'Activity', 'Class']:
                assert subset[column].equals(metadata[column]), f'{target} {model}: {column} mismatch'
            for column in ['Start_Sec', 'End_Sec']:
                np.testing.assert_allclose(subset[column], metadata[column], atol=1e-8)
            matrix = np.zeros((3, 3), dtype=int)
            for true, predicted in zip(subset['Class'], subset['Predicted_Class']):
                matrix[LABELS.index(true), LABELS.index(predicted)] += 1
            saved = pd.read_csv(ROOT / f'cm_{tag}_{target}_auth_counts.csv', index_col=0).loc[LABELS, LABELS].to_numpy()
            np.testing.assert_array_equal(matrix, saved)
            normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                                   out=np.zeros((3, 3)), where=matrix.sum(axis=1, keepdims=True) != 0)
            saved_normalized = pd.read_csv(ROOT / f'cm_{tag}_{target}_auth_normalized.csv', index_col=0).loc[LABELS, LABELS].to_numpy()
            np.testing.assert_allclose(normalized, saved_normalized, atol=1e-10)
            totals[model] += matrix
            report['matrix_checks'].append({'scenario': target, 'model': model, 'counts_match': True,
                                            'row_percentages_match': True, 'empty_row': matrix[2].tolist()})
        report['scenarios'][target] = checked
        print(f'{target}: labels, window alignment and matrix counts verified; saved normalization match={same_mean and same_std}', flush=True)
    for model, tag in [('1D-CNN', '1d_cnn'), ('LSTM', 'lstm')]:
        saved = pd.read_csv(ROOT / f'cm_{tag}_aggregated_counts.csv', index_col=0).loc[LABELS, LABELS].to_numpy()
        np.testing.assert_array_equal(totals[model], saved)
    report['aggregate_counts_match'] = True
    output = ROOT / 'data_audit'
    output.mkdir(exist_ok=True)
    (output / 'code_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
