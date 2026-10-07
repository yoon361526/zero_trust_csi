"""Inspect recording integrity and descriptive CSI statistics; never train models."""

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csi_pipeline import find_matching_files, make_split, read_csi_session


def duplicate_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row['file'])
    return [files for files in groups.values() if len(files) > 1]


def inspect(path):
    path = Path(path)
    timestamps, receivers = [], set()
    lengths = Counter()
    payload_hash = hashlib.sha256()
    malformed_csi = 0
    csi_rows = 0
    header = None
    with path.open(encoding='utf-8', errors='strict') as stream:
        for line_number, line in enumerate(stream):
            if line_number == 0:
                header = line.strip()
            columns = line.rstrip('\r\n').split('\t', 2)
            if len(columns) != 3:
                continue
            try:
                timestamp = int(columns[0])
            except ValueError:
                continue
            timestamps.append(timestamp)
            receivers.add(columns[1])
            if 'CSI_DATA' not in columns[2]:
                continue
            csi_rows += 1
            match = re.search(r'"\[(.*?)\]"', columns[2]) or re.search(r'\[(.*?)\]', columns[2])
            try:
                if match is None:
                    raise ValueError('No payload')
                values = [int(value.strip()) for value in match.group(1).split(',') if value.strip()]
                if len(values) < 2:
                    raise ValueError('Short payload')
            except ValueError:
                malformed_csi += 1
                continue
            lengths[len(values)] += 1
            payload_hash.update((','.join(map(str, values)) + '\n').encode('ascii'))

    session = read_csi_session(str(path))
    times, amplitude = session.elapsed_sec, session.amplitude
    gaps = np.diff(times)
    # Exclude the four-byte invalid leading region and near-zero subcarriers
    # from descriptive relative variability. This is not a presence classifier.
    median_amplitude = np.median(amplitude, axis=0)
    active = median_amplitude > 5
    active[:2] = False
    phase_variability = []
    phase_counts = []
    for start, end in ((0, 60), (60, 120), (120, 180)):
        segment = amplitude[(times >= start) & (times < end)]
        phase_counts.append(len(segment))
        scores = []
        # Local 5-second variation reduces the effect of drift across a minute.
        for begin in range(start, end, 5):
            block = amplitude[(times >= begin) & (times < begin + 5)]
            if len(block) >= 10 and active.any():
                scores.append(float(np.median(np.std(block[:, active], axis=0) /
                                             np.maximum(median_amplitude[active], 1))))
        phase_variability.append(float(np.median(scores)) if scores else None)
    begin_ns, end_ns = min(timestamps), max(timestamps)
    kst = timezone(timedelta(hours=9))
    return {
        'file': f'{path.parent.name}/{path.name}',
        'group': path.parent.name,
        'header_ok': header == 'pi_rx_time_ns\treceiver\traw_data',
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'csi_values_sha256': payload_hash.hexdigest(),
        'start_ns': begin_ns,
        'end_ns': end_ns,
        'start_kst': datetime.fromtimestamp(begin_ns / 1e9, kst).isoformat(timespec='milliseconds'),
        'duration_sec': round((end_ns - begin_ns) / 1e9, 6),
        'raw_records': len(timestamps),
        'non_csi_records': len(timestamps) - csi_rows,
        'malformed_csi_records': malformed_csi,
        'csi_lengths': dict(lengths),
        'receivers': sorted(receivers),
        'timestamp_reversals': sum(b < a for a, b in zip(timestamps, timestamps[1:])),
        'duplicate_timestamps': len(timestamps) - len(set(timestamps)),
        'valid_csi_records': len(times),
        'feature_dim': amplitude.shape[1],
        'finite_amplitude': bool(np.isfinite(amplitude).all()),
        'max_gap_sec': round(float(gaps.max()), 6),
        'gaps_over_half_sec': int((gaps > 0.5).sum()),
        'phase_counts': phase_counts,
        'phase_relative_variability': phase_variability,
    }


def main():
    files = {group: find_matching_files(group) for group in 'ABCD'}
    splits = {target: make_split(files, target) for target in 'ABC'}
    rows = [inspect(path) for paths in files.values() for path in paths]
    overlaps = []
    chronological = sorted(rows, key=lambda row: row['start_ns'])
    for index, left in enumerate(chronological):
        for right in chronological[index + 1:]:
            if right['start_ns'] >= left['end_ns']:
                break
            overlaps.append([left['file'], right['file']])
    summary = {
        'model_training_performed': False,
        'label_identity_verified': False,
        'file_counts': {group: len(paths) for group, paths in files.items()},
        'byte_duplicate_groups': duplicate_groups(rows, 'sha256'),
        'csi_value_duplicate_groups_ignoring_timestamps': duplicate_groups(rows, 'csi_values_sha256'),
        'recording_time_overlaps': overlaps,
        'errors': [row['file'] for row in rows if not row['header_ok'] or not row['finite_amplitude']
                   or row['malformed_csi_records'] or len(row['receivers']) != 1
                   or any(count == 0 for count in row['phase_counts'])],
        'long_gap_recordings': [{key: row[key] for key in ('file', 'max_gap_sec', 'gaps_over_half_sec')}
                                for row in rows if row['gaps_over_half_sec']],
        'groups': {},
        'splits': {target: {key: len(paths) for key, paths in split.items()}
                   for target, split in splits.items()},
        'limitation': 'Folder labels and activity times are supplied by the researcher. CSI statistics cannot establish who was present or prove an empty room.',
    }
    for group in 'ABCD':
        members = [row for row in rows if row['group'] == group]
        summary['groups'][group] = {
            'date_counts': dict(Counter(row['start_kst'][:10] for row in members)),
            'duration_range_sec': [min(row['duration_sec'] for row in members), max(row['duration_sec'] for row in members)],
            'feature_dims': sorted(set(row['feature_dim'] for row in members)),
            'median_relative_variability_by_minute': np.median(
                [row['phase_relative_variability'] for row in members], axis=0).tolist(),
        }
    output = ROOT / 'data_audit'
    output.mkdir(exist_ok=True)
    with (output / 'recordings.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
