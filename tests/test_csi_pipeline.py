import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from csi_pipeline import (
    find_matching_files, make_split, normalize_train_test, parse_csi_to_windows,
    read_csi_session, select_experiment_recordings, validate_unique_recordings,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        read_csi_session.cache_clear()

    def tearDown(self):
        read_csi_session.cache_clear()
        self.directory.cleanup()

    def recording(self, person='A', times=None, prefix=None):
        directory = self.root / person
        directory.mkdir(exist_ok=True)
        path = directory / f'{person}01.txt'
        times = np.arange(3600) / 20 if times is None else times
        origin = 1_700_000_000_000_000_000
        with path.open('w') as stream:
            stream.write('pi_rx_time_ns\treceiver\traw_data\n')
            if prefix is not None:
                stream.write(f'{origin}\tRX1\tpartial packet\n')
            for elapsed in times:
                timestamp = origin + round(float(elapsed) * 1e9)
                stream.write(f'{timestamp}\tRX1\tCSI_DATA,"[3,4]"\n')
        return str(path)

    def test_splits_for_every_authorized_person(self):
        files = {person: [str(self.root / person / f'{person}{i:02d}.txt')
                          for i in range(1, 21)] for person in 'ABCD'}
        for target in 'ABC':
            with self.subTest(target=target):
                split = make_split(files, target)
                self.assertEqual([len(split[name]) for name in
                                  ['train_auth', 'train_unauth', 'train_empty',
                                   'test_auth', 'test_unauth', 'test_empty']],
                                 [16, 16, 16, 4, 4, 4])
                self.assertEqual(split['train_auth'], files[target][:16])
                self.assertEqual(split['test_auth'], files[target][16:20])
                self.assertEqual(split['train_empty'], files['D'][:16])
                self.assertEqual(split['test_empty'], files['D'][16:20])
                for other in set('ABC') - {target}:
                    self.assertEqual([p for p in split['train_unauth']
                                      if Path(p).parent.name == other], files[other][:8])
                    self.assertEqual([p for p in split['test_unauth']
                                      if Path(p).parent.name == other], files[other][16:18])
                train = {p for name, paths in split.items()
                         if name.startswith('train') for p in paths}
                test = {p for name, paths in split.items()
                        if name.startswith('test') for p in paths}
                self.assertFalse(train & test)

    def test_missing_session_number_is_rejected(self):
        files = {person: [str(self.root / person / f'{person}{i:02d}.txt')
                          for i in range(1, 21)] for person in 'ABCD'}
        files['D'][-1] = str(self.root / 'D' / 'D21.txt')
        with self.assertRaises(ValueError):
            make_split(files, 'A')

    def test_random_split_is_reproducible_balanced_and_keeps_files_disjoint(self):
        files = {person: [str(self.root / person / f'{person}{i:02d}.txt')
                          for i in range(1, 21)] for person in 'ABCD'}
        for count in [15, 20]:
            for seed in [0, 1, 42]:
                reference = make_split(files, 'A', count, split_seed=seed)
                self.assertEqual(reference, make_split(files, 'A', count, split_seed=seed))
                for target in 'ABC':
                    split = make_split(files, target, count, split_seed=seed)
                    self.assertEqual(split['train_empty'], reference['train_empty'])
                    self.assertEqual(split['test_empty'], reference['test_empty'])
                    sizes = [len(split[name]) for name in ['train_auth', 'train_unauth',
                             'train_empty', 'test_auth', 'test_unauth', 'test_empty']]
                    self.assertEqual(sizes, [12, 12, 12, 3, 3, 3] if count == 15 else [16, 16, 16, 4, 4, 4])
                    train = {p for name, paths in split.items() if name.startswith('train_') for p in paths}
                    test = {p for name, paths in split.items() if name.startswith('test_') for p in paths}
                    self.assertFalse(train & test)
                    self.assertEqual(len(train) + len(test), sum(sizes))
                    for person in set('ABC') - {target}:
                        person_split = make_split(files, person, count, split_seed=seed)
                        other_train = {p for p in split['train_unauth'] if Path(p).parent.name == person}
                        other_test = {p for p in split['test_unauth'] if Path(p).parent.name == person}
                        self.assertTrue(other_train <= set(person_split['train_auth']))
                        self.assertTrue(other_test <= set(person_split['test_auth']))
        self.assertNotEqual(make_split(files, 'A', split_seed=0), make_split(files, 'A', split_seed=1))
        self.assertNotEqual(make_split(files, 'A', split_seed=0), make_split(files, 'A'))
        for invalid in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                make_split(files, 'A', split_seed=invalid)

    def test_reduced_split_keeps_class_ratio_without_changing_original_lists(self):
        files = {person: [str(self.root / person / f'{person}{i:02d}.txt')
                          for i in range(1, 21)] for person in 'ABCD'}
        selections = select_experiment_recordings(files, 15)
        self.assertEqual([len(paths) for paths in selections.values()], [15] * 4)
        self.assertEqual([len(paths) for paths in files.values()], [20] * 4)
        unauth_test_totals = dict.fromkeys('ABC', 0)
        for target in 'ABC':
            split = make_split(files, target, sessions_per_person=15)
            self.assertEqual([len(split[name]) for name in
                              ['train_auth', 'train_unauth', 'train_empty',
                               'test_auth', 'test_unauth', 'test_empty']],
                             [12, 12, 12, 3, 3, 3])
            self.assertEqual(split['test_empty'], files['D'][12:15])
            train = {p for name, paths in split.items() if name.startswith('train_') for p in paths}
            test = {p for name, paths in split.items() if name.startswith('test_') for p in paths}
            self.assertFalse(train & test)
            self.assertTrue(all(int(Path(p).stem[1:]) <= 15 for p in train | test))
            for person in set('ABC') - {target}:
                self.assertEqual(sum(Path(p).parent.name == person for p in split['train_unauth']), 6)
                unauth_test_totals[person] += sum(Path(p).parent.name == person for p in split['test_unauth'])
        self.assertEqual(unauth_test_totals, {'A': 3, 'B': 3, 'C': 3})

    def test_reduced_split_rejects_missing_number_in_selected_sessions(self):
        files = {person: [str(self.root / person / f'{person}{i:02d}.txt')
                          for i in range(1, 21)] for person in 'ABCD'}
        files['D'][14] = str(self.root / 'D' / 'D16.txt')
        with self.assertRaises(ValueError):
            make_split(files, 'A', sessions_per_person=15)

    def test_copy_across_people_is_rejected(self):
        first = self.recording('A', [0, 0.1])
        other = self.root / 'D'
        other.mkdir()
        copy = other / 'D01.txt'
        copy.write_bytes(Path(first).read_bytes())
        with self.assertRaisesRegex(ValueError, 'A01.txt'):
            validate_unique_recordings({'A': [first], 'D': [str(copy)]})

    def test_folder_search_is_independent_of_working_directory(self):
        directory = self.root / 'A'
        directory.mkdir()
        for number in [20, 2, 1]:
            (directory / f'A{number:02d}.txt').write_text(str(number))
        original = os.getcwd()
        try:
            os.chdir('/tmp')
            paths = find_matching_files('A', str(self.root))
        finally:
            os.chdir(original)
        self.assertEqual([Path(path).name for path in paths], ['A01.txt', 'A02.txt', 'A20.txt'])

    def test_precise_origin_sorting_and_duplicate_timestamp(self):
        path = self.recording(times=[0.2, 0.05, 0.05, 0.1], prefix=True)
        session = read_csi_session(path)
        np.testing.assert_allclose(session.elapsed_sec, [0.05, 0.1, 0.2])
        np.testing.assert_allclose(session.amplitude, 5.0)

    def test_activity_boundaries_and_metadata(self):
        path = self.recording()
        windows, metadata = parse_csi_to_windows([path], 1)
        self.assertEqual(windows.shape, (177, 40, 1))
        np.testing.assert_allclose(windows, 5.0)
        boundaries = {'standing': (0, 60), 'sitting': (60, 120), 'typing': (120, 180)}
        for row in metadata:
            begin, end = boundaries[row['Activity']]
            self.assertGreaterEqual(row['Start_Sec'], begin)
            self.assertLessEqual(row['End_Sec'], end)
            self.assertEqual(row['End_Sec'] - row['Start_Sec'], 2)

    def test_empty_room_has_no_human_activity_labels(self):
        path = self.recording('D')
        windows, metadata = parse_csi_to_windows([path], 1, is_empty=True)
        self.assertEqual(windows.shape, (179, 40, 1))
        self.assertEqual({row['Activity'] for row in metadata}, {'empty'})

    def test_resampling_uses_time_instead_of_packet_count(self):
        path = self.recording(times=np.arange(1800) / 10 + 0.003, prefix=True)
        windows, metadata = parse_csi_to_windows([path], 1)
        self.assertEqual(windows.shape[1:], (40, 1))
        self.assertGreater(len(windows), 160)
        standing = [row for row in metadata if row['Activity'] == 'standing']
        self.assertAlmostEqual(standing[0]['Start_Sec'], 0.05)
        self.assertAlmostEqual(standing[1]['Start_Sec'] - standing[0]['Start_Sec'], 1)

    def test_long_gap_is_not_interpolated_into_windows(self):
        times = np.arange(3600) / 20
        times = times[(times <= 10) | (times >= 12)]
        path = self.recording(times=times)
        _, metadata = parse_csi_to_windows([path], 1)
        for row in metadata:
            self.assertFalse(row['Start_Sec'] < 12 and row['End_Sec'] > 10.05)

    def test_normalization_does_not_fit_test_data(self):
        train = np.array([[[1.0], [3.0]]], dtype=np.float32)
        test = np.array([[[100.0], [100.0]]], dtype=np.float32)
        normalized_train, normalized_test, mean, std = normalize_train_test(train, test)
        np.testing.assert_allclose(normalized_train, [[[-1], [1]]])
        np.testing.assert_allclose(normalized_test, [[[98], [98]]])
        np.testing.assert_allclose(mean, [2])
        np.testing.assert_allclose(std, [1])

    def test_absolute_recording_date_does_not_change_model_windows(self):
        path = self.recording('D', np.arange(201) / 20)
        before, before_metadata = parse_csi_to_windows([path], 1, is_empty=True)
        lines = Path(path).read_text().splitlines()
        shifted = [lines[0]]
        for line in lines[1:]:
            timestamp, receiver, payload = line.split('\t', 2)
            shifted.append(f'{int(timestamp) + 86_400_000_000_000}\t{receiver}\t{payload}')
        Path(path).write_text('\n'.join(shifted) + '\n')
        read_csi_session.cache_clear()
        after, after_metadata = parse_csi_to_windows([path], 1, is_empty=True)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before_metadata, after_metadata)

    def standard_packet(self, payload, first_word=1, declared_length=None):
        fields = ['CSI_DATA', '1', '1a:00:00:00:00:00', '-55', '11', '1', '0',
                  '1', '1', '1', '0', '0', '0', '0', '-96', '0', '11', '2',
                  '123456', '0', '47', '1',
                  str(len(payload) if declared_length is None else declared_length), str(first_word)]
        return ','.join(fields) + ',"[' + ','.join(map(str, payload)) + ']"'

    def packet_log(self, packets):
        path = self.root / 'packets.txt'
        with path.open('w') as stream:
            stream.write('pi_rx_time_ns\treceiver\traw_data\n')
            for index, packet in enumerate(packets):
                stream.write(f'{1_700_000_000_000_000_000 + index * 50_000_000}\tRX1\t{packet}\n')
        return str(path)

    def test_hardware_invalid_first_four_bytes_cannot_become_features(self):
        path = self.packet_log([
            self.standard_packet([100, -100, 60, 40, 3, 4]),
            self.standard_packet([-128, 127, 1, 2, 3, 4]),
        ])
        session = read_csi_session(path)
        self.assertEqual(session.amplitude.shape, (2, 3))
        np.testing.assert_allclose(session.amplitude, [[0, 0, 5], [0, 0, 5]])

    def test_valid_first_word_is_retained_when_flag_is_zero(self):
        path = self.packet_log([self.standard_packet([3, 4, 6, 8, 5, 12], first_word=0)])
        np.testing.assert_allclose(read_csi_session(path).amplitude, [[5, 10, 13]])

    def test_declared_length_mismatch_is_rejected_without_misaligning_rows(self):
        path = self.packet_log([
            self.standard_packet([100, 0, 100, 0, 30, 40], declared_length=8),
            self.standard_packet([100, 0, 100, 0, 3, 4]),
        ])
        session = read_csi_session(path)
        np.testing.assert_allclose(session.elapsed_sec, [0.05])
        np.testing.assert_allclose(session.amplitude, [[0, 0, 5]])


if __name__ == '__main__':
    unittest.main()
