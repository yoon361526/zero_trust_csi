"""Session-level splitting and timestamp-based CSI preprocessing."""

import glob
import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PREPROCESSING_VERSION = 2
ACTIVITY_SEGMENTS = (
    (0.0, 60.0, 'standing'),
    (60.0, 120.0, 'sitting'),
    (120.0, 180.0, 'typing'),
)
CLASS_NAMES = ('Auth', 'Unauth', 'Empty')
PERSON_TRAIN_N = 16
PERSON_TEST_N = 4
UNAUTH_TRAIN_PER_PERSON = 8
UNAUTH_TEST_PER_PERSON = 2
EMPTY_TRAIN_N = 16
EMPTY_TEST_N = 4
REDUCED_TEST_COUNTS = {
    'A': {'B': 1, 'C': 2},
    'B': {'A': 2, 'C': 1},
    'C': {'A': 1, 'B': 2},
}


def find_matching_files(prefix, search_dir=DATA_DIR):
    session_dir = os.path.join(search_dir, prefix)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f'데이터 폴더를 찾을 수 없습니다: {session_dir}')
    pattern = re.compile(
        rf'(?i)(?:^|[^A-Za-z0-9]){re.escape(prefix)}0*(\d+)(?:[^0-9]|$)'
    )
    sessions = {}
    for path in sorted(glob.glob(os.path.join(session_dir, '*.txt'))):
        match = pattern.search(os.path.basename(path))
        if match:
            number = int(match.group(1))
            if number in sessions:
                raise ValueError(f'{prefix}{number:02d} 세션 파일이 중복됩니다.')
            sessions[number] = path
    return [sessions[number] for number in sorted(sessions)]


def validate_unique_recordings(files_all):
    """Reject copied recordings before they can contaminate labels or splits."""
    hashes = {}
    duplicates = []
    for files in files_all.values():
        for path in files:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            key = digest.hexdigest()
            if key in hashes:
                duplicates.append((hashes[key], path))
            else:
                hashes[key] = path
    if duplicates:
        pairs = '\n'.join(f'  {left} = {right}' for left, right in duplicates)
        raise ValueError(
            '내용이 동일한 측정 파일이 있습니다. 사람/빈방 라벨과 원본 데이터를 '
            f'확인한 뒤 다시 실행하세요.\n{pairs}'
        )


def select_experiment_recordings(files_all, sessions_per_person=20):
    """Keep the first numbered sessions; never delete or modify recordings."""
    if sessions_per_person not in (15, 20):
        raise ValueError('실험은 대상별 15개 또는 20개 세션을 지원합니다.')
    selected = {}
    for person in 'ABCD':
        paths = files_all[person]
        if len(paths) < sessions_per_person:
            raise ValueError(f'{person}: 최소 {sessions_per_person}개 세션이 필요합니다.')
        selected[person] = (paths[:15] if sessions_per_person == 15 else list(paths))
    return selected


def make_split(files_all, target, sessions_per_person=20, split_seed=None):
    """Split whole recordings; an optional seed permutes each person's order."""
    if target not in ('A', 'B', 'C'):
        raise ValueError(f'알 수 없는 인가자: {target}')
    files_all = select_experiment_recordings(files_all, sessions_per_person)
    person_train = 12 if sessions_per_person == 15 else PERSON_TRAIN_N
    person_test = 3 if sessions_per_person == 15 else PERSON_TEST_N
    unauth_train = 6 if sessions_per_person == 15 else UNAUTH_TRAIN_PER_PERSON
    empty_train = 12 if sessions_per_person == 15 else EMPTY_TRAIN_N
    empty_test = 3 if sessions_per_person == 15 else EMPTY_TEST_N
    for person in ('A', 'B', 'C', 'D'):
        required = sessions_per_person
        if len(files_all[person]) != required:
            raise ValueError(f'{person}: 정확히 {required}개 세션이 필요합니다.')
        pattern = re.compile(
            rf'(?i)(?:^|[^A-Za-z0-9]){person}0*(\d+)(?:[^0-9]|$)'
        )
        numbers = []
        for path in files_all[person]:
            match = pattern.search(os.path.basename(path))
            numbers.append(int(match.group(1)) if match else -1)
        if numbers != list(range(1, required + 1)):
            raise ValueError(f'{person}01~{person}{required:02d} 세션을 확인하세요.')

    if split_seed is not None:
        if (not isinstance(split_seed, (int, np.integer))
                or isinstance(split_seed, bool) or split_seed < 0):
            raise ValueError('분할 seed는 0 이상의 정수여야 합니다.')
        # One permutation per person is shared by all authorized-person scenarios.
        # Hold out the last 20% BEFORE choosing the smaller unauthorized subset.
        # Overlapping windows from a recording always stay in one partition.
        files_all = {
            person: [paths[int(i)] for i in np.random.default_rng(
                np.random.SeedSequence([int(split_seed), index])).permutation(len(paths))]
            for index, (person, paths) in enumerate(files_all.items())
        }

    others = [person for person in ('A', 'B', 'C') if person != target]
    test_counts = (REDUCED_TEST_COUNTS[target] if sessions_per_person == 15
                   else {person: UNAUTH_TEST_PER_PERSON for person in others})
    split = {
        'train_auth': files_all[target][:person_train],
        'test_auth': files_all[target][person_train:person_train + person_test],
        'train_unauth': [path for person in others
                         for path in files_all[person][:unauth_train]],
        'test_unauth': [path for person in others
                        for path in files_all[person][person_train:
                                                     person_train + test_counts[person]]],
        'train_empty': files_all['D'][:empty_train],
        'test_empty': files_all['D'][empty_train:empty_train + empty_test],
    }
    train_files = {path for name, paths in split.items()
                   if name.startswith('train_') for path in paths}
    test_files = {path for name, paths in split.items()
                  if name.startswith('test_') for path in paths}
    if train_files & test_files:
        raise ValueError('같은 원본 파일이 학습과 테스트에 포함되어 있습니다.')
    return split


@dataclass(frozen=True)
class CSISession:
    elapsed_sec: np.ndarray
    amplitude: np.ndarray
    receiver: str


@lru_cache(maxsize=128)
def read_csi_session(filepath):
    """Read pi_rx_time_ns / receiver / raw_data logs; keep the capture origin."""
    records = []
    origin_ns = None
    with open(filepath, encoding='utf-8', errors='ignore') as stream:
        for line in stream:
            columns = line.rstrip('\r\n').split('\t', 2)
            if len(columns) != 3:
                continue
            try:
                timestamp = int(columns[0])
            except ValueError:
                continue
            origin_ns = timestamp if origin_ns is None else min(origin_ns, timestamp)
            if 'CSI_DATA' not in columns[2]:
                continue
            match = re.search(r'"\[(.*?)\]"', columns[2]) or re.search(r'\[(.*?)\]', columns[2])
            if not match:
                continue
            try:
                values = [int(value.strip()) for value in match.group(1).split(',')
                          if value.strip()]
            except ValueError:
                continue
            # Espressif's standard log has 24 fields before the payload:
            # ... len, first_word, "[data]". first_word=1 declares the first
            # four signed bytes invalid. Preserve feature positions by masking
            # those two amplitude features, rather than shifting subcarriers.
            header = columns[2][:match.start()].rstrip(', ').split(',')
            invalid_first_word = False
            if len(header) == 24:
                try:
                    declared_length, first_word = int(header[22]), int(header[23])
                except ValueError:
                    continue
                if (declared_length != len(values) or declared_length % 2
                        or first_word not in (0, 1)):
                    continue
                invalid_first_word = first_word == 1
                if invalid_first_word and len(values) <= 4:
                    continue
            elif header != ['CSI_DATA']:
                # Minimal CSI_DATA,"[data]" logs have no hardware flag.
                # A partial standard header must not be interpreted as valid.
                continue
            if len(values) >= 2:
                records.append((timestamp, columns[1], values, invalid_first_word))
    if not records:
        raise ValueError(f'타임스탬프가 있는 유효한 CSI_DATA가 없습니다: {filepath}')

    lengths, counts = np.unique([len(row[2]) for row in records], return_counts=True)
    common_length = int(lengths[np.argmax(counts)])
    records = [row for row in records if len(row[2]) == common_length]
    receivers = {row[1] for row in records}
    if len(receivers) != 1:
        raise ValueError(f'여러 수신기의 CSI가 섞여 있습니다: {filepath}: {receivers}')
    records.sort(key=lambda row: row[0])
    timestamps = np.asarray([row[0] for row in records], dtype=np.int64)
    values = np.asarray([row[2] for row in records], dtype=np.float32)
    timestamps, unique_indices = np.unique(timestamps, return_index=True)
    values = values[unique_indices, :common_length - common_length % 2]
    invalid_first_words = np.asarray([row[3] for row in records], dtype=bool)[unique_indices]
    values[invalid_first_words, :4] = 0
    amplitude = np.hypot(values[:, 0::2], values[:, 1::2]).astype(np.float32)
    elapsed = (timestamps - origin_ns).astype(np.float64) / 1e9
    return CSISession(elapsed, amplitude, next(iter(receivers)))


def determine_global_feature_dim(file_groups):
    """Use training recordings only to choose the amplitude dimension."""
    dims = [read_csi_session(path).amplitude.shape[1]
            for path in sorted({p for group in file_groups for p in group})]
    if not dims:
        raise ValueError('학습 CSI 파일이 없습니다.')
    values, counts = np.unique(dims, return_counts=True)
    return int(values[np.argmax(counts)])


def parse_csi_to_windows(file_list, expected_feature_dim, window_sec=2,
                         stride_sec=1, target_fps=20, max_gap_sec=0.5,
                         is_empty=False):
    """Resample within activity boundaries; never extrapolate or bridge long gaps."""
    window_frames = int(round(window_sec * target_fps))
    stride_frames = int(round(stride_sec * target_fps))
    if (window_frames < 1 or stride_frames < 1 or target_fps <= 0 or max_gap_sec <= 0
            or not np.isclose(window_frames, window_sec * target_fps)
            or not np.isclose(stride_frames, stride_sec * target_fps)):
        raise ValueError('윈도우/간격은 양의 정수 프레임에 대응해야 합니다.')
    segments = ((0.0, 180.0, 'empty'),) if is_empty else ACTIVITY_SEGMENTS
    windows, metadata = [], []
    for path in file_list:
        session = read_csi_session(path)
        if session.amplitude.shape[1] != expected_feature_dim:
            raise ValueError(f'CSI 특징 차원이 다릅니다: {path}: '
                             f'{session.amplitude.shape[1]} != {expected_feature_dim}')
        for begin, end, activity in segments:
            mask = (session.elapsed_sec >= begin) & (session.elapsed_sec < end)
            times, amplitude = session.elapsed_sec[mask], session.amplitude[mask]
            first_window = len(windows)
            if len(times) >= 2:
                first_frame = int(np.ceil(times[0] * target_fps - 1e-9))
                last_frame = int(np.floor(times[-1] * target_fps + 1e-9))
                grid = np.arange(first_frame, last_frame + 1, dtype=np.float64) / target_fps
                if len(grid) >= window_frames:
                    right = np.minimum(np.searchsorted(times, grid), len(times) - 1)
                    left = np.maximum(right - 1, 0)
                    exact = np.isclose(grid, times[right], rtol=0, atol=1e-9)
                    supported = exact | (times[right] - times[left] <= max_gap_sec + 1e-9)
                    resampled = np.column_stack([
                        np.interp(grid, times, amplitude[:, feature])
                        for feature in range(expected_feature_dim)
                    ]).astype(np.float32)
                    for start in range(0, len(grid) - window_frames + 1, stride_frames):
                        stop = start + window_frames
                        if not supported[start:stop].all():
                            continue
                        windows.append(resampled[start:stop])
                        metadata.append({
                            'Person': os.path.basename(os.path.dirname(path)).upper(),
                            'Session': os.path.basename(path),
                            'Source_File': os.path.abspath(path),
                            'Receiver': session.receiver,
                            'Activity': activity,
                            'Start_Sec': round(float(grid[start]), 6),
                            'End_Sec': round(float(grid[start] + window_sec), 6),
                        })
            if len(windows) == first_window:
                raise ValueError(f'{path}: {activity} 구간에 유효한 윈도우가 없습니다.')
    if not windows:
        return np.empty((0, window_frames, expected_feature_dim), np.float32), []
    return np.asarray(windows, dtype=np.float32), metadata


def normalize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.std(X_train, axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def prepare_scenario_data(split, window_sec=2, stride_sec=1, target_fps=20,
                          max_gap_sec=0.5, seed=42):
    feature_dim = determine_global_feature_dim([
        split['train_auth'], split['train_unauth'], split['train_empty'],
    ])
    groups, group_metadata = {}, {}
    for name, paths in split.items():
        groups[name], group_metadata[name] = parse_csi_to_windows(
            paths, feature_dim, window_sec, stride_sec, target_fps,
            max_gap_sec, is_empty=name.endswith('_empty'),
        )
    data = {'feature_dim': feature_dim}
    for partition in ('train', 'test'):
        data[f'X_{partition}'] = np.concatenate([
            groups[f'{partition}_{role}'] for role in ('auth', 'unauth', 'empty')
        ])
        data[f'y_{partition}'] = np.concatenate([
            np.full(len(groups[f'{partition}_{role}']), label, dtype=np.int64)
            for label, role in enumerate(('auth', 'unauth', 'empty'))
        ])
        data[f'{partition}_metadata'] = [
            {**row, 'Class': CLASS_NAMES[label]}
            for label, role in enumerate(('auth', 'unauth', 'empty'))
            for row in group_metadata[f'{partition}_{role}']
        ]
    data['X_train'], data['X_test'], data['mean'], data['std'] = normalize_train_test(
        data['X_train'], data['X_test'],
    )
    permutation = np.random.default_rng(seed).permutation(len(data['X_train']))
    data['X_train'] = data['X_train'][permutation]
    data['y_train'] = data['y_train'][permutation]
    data['train_metadata'] = [data['train_metadata'][int(i)] for i in permutation]
    return data
