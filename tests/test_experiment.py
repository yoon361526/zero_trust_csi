import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import cha_gpt


class ExperimentTests(unittest.TestCase):
    def test_model_archive_restores_probabilities_without_training(self):
        with tempfile.TemporaryDirectory() as directory, cha_gpt.tf.device('/CPU:0'):
            model = cha_gpt.build_1d_cnn((40, 2), filters=32, kernel_size=3)
            features = np.random.default_rng(42).normal(size=(2, 40, 2)).astype(np.float32)
            before = model(features, training=False).numpy()
            path = Path(directory, 'example.keras')
            model.save(path)
            restored = cha_gpt.tf.keras.models.load_model(path, compile=False)
            after = restored(features, training=False).numpy()
            np.testing.assert_allclose(before, after, atol=1e-6)
            np.testing.assert_allclose(after.sum(axis=1), 1, atol=1e-6)
        cha_gpt.tf.keras.backend.clear_session()

    def test_activity_metrics_keep_empty_separate(self):
        actual = np.array([0, 1, 0, 1, 0, 1, 2])
        predicted = np.array([0, 2, 1, 1, 0, 1, 0])
        metadata = [{'Activity': activity} for activity in
                    ['standing', 'standing', 'sitting', 'sitting', 'typing', 'typing', 'empty']]
        rows = cha_gpt.evaluate_activities(actual, predicted, metadata, 'A', '1D-CNN')
        self.assertEqual([row['Accuracy'] for row in rows], [0.5, 0.5, 1.0, 0.0])
        self.assertEqual(rows[0]['Evaluated_Classes'], 'Auth|Unauth')
        self.assertEqual(rows[-1]['Evaluated_Classes'], 'Empty')
        self.assertEqual(sum(row['Support'] for row in rows), len(actual))

    def test_prepare_only_never_builds_or_trains_models_or_saves_results(self):
        split = {name: [f'/data/{name}.txt'] for name in
                 ['train_auth', 'train_unauth', 'train_empty',
                  'test_auth', 'test_unauth', 'test_empty']}
        data = {f'X_{partition}': np.zeros((3, 40, 2), dtype=np.float32)
                for partition in ['train', 'test']}
        data.update({f'y_{partition}': np.array([0, 1, 2]) for partition in ['train', 'test']})
        with patch.object(cha_gpt, 'find_matching_files', return_value=['session']), \
                patch.object(cha_gpt, 'make_split', return_value=split), \
                patch.object(cha_gpt, 'validate_unique_recordings'), \
                patch.object(cha_gpt, 'prepare_scenario_data', return_value=data) as prepare, \
                patch.object(cha_gpt, 'build_1d_cnn') as cnn, \
                patch.object(cha_gpt, 'build_lstm') as lstm, \
                patch.object(cha_gpt.np, 'savez') as save, \
                patch.object(cha_gpt.pd.DataFrame, 'to_csv') as csv, \
                contextlib.redirect_stdout(io.StringIO()):
            cha_gpt.main(prepare_only=True)
            self.assertEqual(prepare.call_count, 3)
            cnn.assert_not_called()
            lstm.assert_not_called()
            save.assert_not_called()
            csv.assert_not_called()

    def test_duplicate_recordings_abort_before_preprocessing(self):
        with patch.object(cha_gpt, 'find_matching_files', return_value=['session']), \
                patch.object(cha_gpt, 'make_split', return_value={}), \
                patch.object(cha_gpt, 'validate_unique_recordings', side_effect=ValueError('duplicate')), \
                patch.object(cha_gpt, 'prepare_scenario_data') as prepare, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                cha_gpt.main(prepare_only=True)
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
