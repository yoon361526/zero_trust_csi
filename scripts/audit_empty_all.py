"""Audit every D recording and saved splits/results without fitting a model."""

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import (
    PREPROCESSING_VERSION, find_matching_files, make_split,
    parse_csi_to_windows, read_csi_session,
)
from audit_data import duplicate_groups, inspect


def packet_metadata(path):
    categories = {name: Counter() for name in (
        'mac', 'channel', 'secondary_channel', 'bandwidth', 'sig_mode',
        'mcs', 'rate', 'stbc', 'sgi', 'rx_state', 'len', 'first_word',
    )}
    positions = dict(mac=2, channel=16, secondary_channel=17, bandwidth=7,
                     sig_mode=5, mcs=6, rate=4, stbc=11, sgi=13,
                     rx_state=21, len=22, first_word=23)
    rejected = Counter()
    numeric = {'rssi': [], 'noise_floor': []}
    packet_ids, local_times = [], []
    payload_count = valid_headers = saturated = values_count = 0
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            columns = line.rstrip('\r\n').split('\t', 2)
            if len(columns) != 3 or 'CSI_DATA' not in columns[2]:
                continue
            payload_count += 1
            match = re.search(r'"\[(.*?)\]"', columns[2]) or re.search(r'\[(.*?)\]', columns[2])
            if match is None:
                rejected['missing_payload'] += 1
                continue
            try:
                values = [int(item.strip()) for item in match.group(1).split(',') if item.strip()]
            except ValueError:
                rejected['non_integer_payload'] += 1
                continue
            header = columns[2][:match.start()].rstrip(', ').split(',')
            if len(header) != 24:
                rejected['partial_header'] += 1
                continue
            try:
                declared_length, flag = int(header[22]), int(header[23])
            except ValueError:
                rejected['invalid_length_or_flag'] += 1
                continue
            if declared_length != len(values) or declared_length % 2:
                rejected['length_mismatch_or_odd'] += 1
                continue
            if flag not in (0, 1) or len(values) <= (4 if flag else 1):
                rejected['invalid_flag_or_short_payload'] += 1
                continue
            valid_headers += 1
            for name, index in positions.items():
                categories[name][header[index]] += 1
            for name, index in [('rssi', 3), ('noise_floor', 14)]:
                numeric[name].append(float(header[index]))
            packet_ids.append(int(header[1]))
            local_times.append(int(header[18]))
            useful_values = values[4:] if flag else values
            saturated += sum(abs(value) >= 127 for value in useful_values)
            values_count += len(useful_values)
    return {
        'csi_rows': payload_count, 'complete_valid_packets': valid_headers,
        'rejected_csi_rows': sum(rejected.values()), 'rejection_reasons': dict(rejected),
        'metadata_counts': {name: dict(counts) for name, counts in categories.items()},
        'numeric_metadata': {name: {
            'min': min(values), 'median': float(np.median(values)), 'max': max(values),
        } for name, values in numeric.items()},
        'packet_id_backward_steps': sum(b < a for a, b in zip(packet_ids, packet_ids[1:])),
        'device_clock_backward_steps': sum(b < a for a, b in zip(local_times, local_times[1:])),
        'near_limit_signed_byte_fraction': saturated / max(values_count, 1),
    }


def pooled_packet_scale(paths):
    """Descriptive reference only: pooled training-packet SD, not model normalization."""
    count = 0
    total = squares = None
    for path in paths:
        amplitude = read_csi_session(path).amplitude[:, 2:].astype(np.float64)
        count += len(amplitude)
        sums = amplitude.sum(axis=0)
        sum_squares = (amplitude ** 2).sum(axis=0)
        total = sums if total is None else total + sums
        squares = sum_squares if squares is None else squares + sum_squares
    scale = np.sqrt(np.maximum(squares / count - (total / count) ** 2, 0))
    scale[scale < 1e-8] = 1
    return scale


def result_provenance(files, predictions, basics):
    checks = []
    for path in files['D']:
        name = Path(path).name
        _, info = parse_csi_to_windows([path], 192, is_empty=True)
        current = pd.DataFrame(info)
        for target in 'ABC':
            for model in ('1D-CNN', 'LSTM'):
                saved = predictions[(predictions['Scenario'] == f'{target} Auth') &
                                    (predictions['Model'] == model) &
                                    (predictions['Person'] == 'D') &
                                    (predictions['Session'] == name)]
                if saved.empty:
                    continue
                matches = len(saved) == len(current)
                if matches:
                    matches = all(np.allclose(saved[key].to_numpy(), current[key].to_numpy(),
                                              rtol=0, atol=1e-6)
                                  for key in ('Start_Sec', 'End_Sec'))
                checks.append({'file': name, 'scenario': target, 'model': model,
                               'current_window_times_match_saved': bool(matches)})
    previous_changes = []
    previous = ROOT / 'data_audit/recordings.csv'
    if previous.exists():
        for row in pd.read_csv(previous).to_dict(orient='records'):
            if row['group'] == 'D' and row['sha256'] != basics[row['file']]['sha256']:
                previous_changes.append({'file': row['file'], 'previous_start_kst': row['start_kst'],
                                         'current_start_kst': basics[row['file']]['start_kst']})
    return {'saved_window_metadata_checks': checks,
            'changed_since_previous_integrity_report': previous_changes,
            'saved_prediction_input_hash_available': False,
            'limitation': 'Matching names/window times does not prove the same CSI contents were used. Existing prediction files do not contain source hashes.'}


def main():
    output = ROOT / 'data_audit'
    output.mkdir(exist_ok=True)
    files = {person: find_matching_files(person) for person in 'ABCD'}
    # Re-read all 80 originals; comparisons do not trust an older report.
    basic_rows = [inspect(path) for paths in files.values() for path in paths]
    basics = {row['file']: row for row in basic_rows}
    profiles = {path: np.median(read_csi_session(path).amplitude[:, 2:], axis=0)
                for paths in files.values() for path in paths}
    saved_split = pd.read_csv(ROOT / 'csi_session_split.csv')
    predictions = pd.read_csv(ROOT / 'csi_test_window_predictions.csv')
    rows, details, split_checks = [], {}, {}
    d_profiles = np.stack([profiles[path] for path in files['D']])
    raw_distances = np.sqrt(np.mean((d_profiles[:, None, :] - d_profiles[None, :, :]) ** 2, axis=2))
    assert np.allclose(raw_distances, raw_distances.T) and np.allclose(np.diag(raw_distances), 0)
    nearest_by_scenario = {}
    for target in 'ABC':
        split = make_split(files, target)
        d_train = {Path(path).name for path in split['train_empty']}
        d_test = {Path(path).name for path in split['test_empty']}
        saved = saved_split[(saved_split['Scenario'] == f'{target} Auth') & (saved_split['Person'] == 'D')]
        assert set(saved[saved['Partition'] == 'train']['Session']) == d_train
        assert set(saved[saved['Partition'] == 'test']['Session']) == d_test
        assert (saved['Class'] == 'Empty').all() and len(saved) == 20
        split_checks[target] = {'train': sorted(d_train), 'test': sorted(d_test),
                                'overlap': sorted(d_train & d_test), 'saved_split_matches': True}
        training = [path for key, paths in split.items() if key.startswith('train_') for path in paths]
        scale = pooled_packet_scale(training)
        distances = np.sqrt(np.mean(((d_profiles[:, None, :] - np.stack(
            [profiles[path] for path in training])[None, :, :]) / scale) ** 2, axis=2))
        scenario_rows = []
        for index, path in enumerate(files['D']):
            references = []
            for person in 'ABCD':
                candidates = [j for j, ref in enumerate(training)
                              if Path(ref).parent.name == person and ref != path]
                closest = min(candidates, key=lambda j: distances[index, j])
                references.append({'group': person, 'nearest_training_file': str(Path(training[closest]).relative_to(ROOT / 'data')),
                                   'distance': float(distances[index, closest])})
            scenario_rows.append({'file': Path(path).name, 'references': references})
        nearest_by_scenario[target] = scenario_rows
    for index, path in enumerate(files['D']):
        name = Path(path).name
        basic = basics[f'D/{name}']
        session = read_csi_session(path)
        amplitude = session.amplitude[:, 2:]
        metadata = packet_metadata(path)
        windows, window_info = parse_csi_to_windows([path], 192, is_empty=True)
        assert len(windows) == len(window_info) and np.isfinite(windows).all()
        minute_profiles = [np.median(amplitude[(session.elapsed_sec >= begin) &
                                                (session.elapsed_sec < begin + 60)], axis=0)
                           for begin in (0, 60, 120)]
        minute_drift = max(float(np.sqrt(np.mean((a - b) ** 2)))
                           for i, a in enumerate(minute_profiles) for b in minute_profiles[i+1:])
        other = np.argsort(raw_distances[index])[1]
        nearest = nearest_by_scenario['A'][index]['references'][-1]
        row = {
            'file': name, 'partition': 'train' if index < 16 else 'test',
            'start_kst': basic['start_kst'], 'duration_sec': basic['duration_sec'],
            'valid_packets': len(amplitude), 'rejected_csi_rows': metadata['rejected_csi_rows'],
            'max_gap_sec': basic['max_gap_sec'], 'gaps_over_half_sec': basic['gaps_over_half_sec'],
            'windows': len(windows), 'median_rssi': metadata['numeric_metadata']['rssi']['median'],
            'median_noise_floor': metadata['numeric_metadata']['noise_floor']['median'],
            'mean_median_profile': float(np.mean(profiles[path])),
            'minute1_variability': basic['phase_relative_variability'][0],
            'minute2_variability': basic['phase_relative_variability'][1],
            'minute3_variability': basic['phase_relative_variability'][2],
            'max_minute_profile_drift_rms': minute_drift,
            'nearest_other_D': Path(files['D'][other]).name,
            'nearest_other_D_raw_profile_rms': float(raw_distances[index, other]),
            'nearest_training_D_A': nearest['nearest_training_file'],
            'nearest_training_D_scaled_distance_A': nearest['distance'],
            'finite': basic['finite_amplitude'], 'timestamp_reversals': basic['timestamp_reversals'],
            'duplicate_timestamps': basic['duplicate_timestamps'],
            'receiver': session.receiver, 'feature_dim': session.amplitude.shape[1],
        }
        for target in 'ABC':
            for model in ('1D-CNN', 'LSTM'):
                saved = predictions[(predictions['Scenario'] == f'{target} Auth') &
                                    (predictions['Model'] == model) & (predictions['Person'] == 'D') &
                                    (predictions['Session'] == name)]
                key = f'{target}_{model}_saved_empty_recall'
                row[key] = float((saved['Predicted_Class'] == 'Empty').mean()) if len(saved) else None
                if len(saved):
                    assert (saved['Class'] == 'Empty').all()
                    assert len(saved) == len(windows)
        rows.append(row)
        details[name] = {'integrity': basic, 'packets': metadata,
                         'adjacent_identical_amplitude_fraction': float(np.mean(
                             np.all(amplitude[1:] == amplitude[:-1], axis=1)))}
    assert len(rows) == 20 and sum(row['partition'] == 'train' for row in rows) == 16
    # Detect any original changing during the audit.
    for paths in files.values():
        for path in paths:
            key = f'{Path(path).parent.name}/{Path(path).name}'
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == basics[key]['sha256']
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'D_all_files_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(raw_distances, index=[Path(path).name for path in files['D']],
                 columns=[Path(path).name for path in files['D']]).to_csv(output / 'D_profile_distances.csv')
    report = {
        'checked_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'model_training_performed': False, 'preprocessing_version': PREPROCESSING_VERSION,
        'originals_unchanged': True, 'all_80_byte_duplicate_groups': duplicate_groups(basic_rows, 'sha256'),
        'all_80_payload_duplicate_groups': duplicate_groups(basic_rows, 'csi_values_sha256'),
        'D_rows': rows, 'D_details': details, 'split_checks': split_checks,
        'nearest_training_profiles_by_scenario': nearest_by_scenario,
        'result_provenance': result_provenance(files, predictions, basics),
        'distance_definition': 'Median amplitude profile RMS; scaled references use current pooled training-packet SD, not saved model normalization. Own file excluded for training D.',
        'limitations': ['No trained models were fitted or evaluated. Training D predictions were not saved in the existing run, so training recall is unavailable.',
                       'Saved test predictions are from before preprocessing_version=2; this audit does not recalculate model predictions.',
                       'Timestamps are logger clock values; their agreement with actual measurement dates is not independently verified.',
                       'Profile similarity and signal variability cannot establish person identity or prove an empty room.'],
    }
    (output / 'D_all_files_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    labels = [f'D{i:02d} {"train" if i <= 16 else "test"}' for i in range(1, 21)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    image = axes[0].imshow(d_profiles, aspect='auto', cmap='viridis')
    axes[0].set(title='All D recordings: median amplitude profiles', xlabel='Feature index (first two excluded)',
                yticks=np.arange(20), yticklabels=labels)
    axes[0].axhline(15.5, color='red', linewidth=1)
    fig.colorbar(image, ax=axes[0], label='CSI amplitude')
    image = axes[1].imshow(raw_distances, cmap='magma')
    axes[1].set(title='Between-file profile RMS distance (descriptive)',
                xticks=np.arange(20), xticklabels=[f'{i:02d}' for i in range(1,21)],
                yticks=np.arange(20), yticklabels=[f'D{i:02d}' for i in range(1,21)])
    axes[1].axhline(15.5, color='cyan', linewidth=1)
    axes[1].axvline(15.5, color='cyan', linewidth=1)
    fig.colorbar(image, ax=axes[1], label='RMS amplitude difference')
    fig.savefig(output / 'D_all_profiles.png', dpi=160)
    plt.close(fig)
    print(frame.to_string(index=False))
    print('Byte duplicate groups:', report['all_80_byte_duplicate_groups'])
    print('Payload duplicate groups:', report['all_80_payload_duplicate_groups'])
    print('Outputs:', output / 'D_all_files_audit.json', output / 'D_all_profiles.png')


if __name__ == '__main__':
    if '--check-result-provenance-only' in sys.argv:
        report_path = ROOT / 'data_audit/D_all_files_audit.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        basics = {f'D/{name}': detail['integrity'] for name, detail in report['D_details'].items()}
        files = {'D': find_matching_files('D')}
        for path in files['D']:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == basics[f'D/{Path(path).name}']['sha256']
        report['result_provenance'] = result_provenance(
            files, pd.read_csv(ROOT / 'csi_test_window_predictions.csv'), basics)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report['result_provenance'], ensure_ascii=False, indent=2))
    else:
        main()
