import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import cha_gpt
import evaluate_optuna as evaluation
import tune_optuna as tuning
from csi_pipeline import make_split


class ConfusionReportingTests(unittest.TestCase):
    def files(self):
        return {person: [f'/data/{person}/{person}{i:02d}.txt' for i in range(1, 21)]
                for person in 'ABCD'}

    def best(self):
        return {'authorized_person': 'A', 'model': 'cnn', 'best_trial': 3,
                'protocol_sha256': 'matching-protocol', 'max_epochs': 60,
                'params': {'window_sec': 2, 'batch_size': 32, 'dropout': 0.3,
                           'learning_rate': 0.001, 'filters': 32, 'kernel_size': 3},
                'diagnostics': {'best_epoch': 2, 'epochs_run': 7}}

    def data(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        return {'X_train': np.zeros((9, 40, 2), dtype=np.float32),
                'y_train': np.repeat([0, 1, 2], 3),
                'train_metadata': [{'Activity': 'empty' if label == 2 else 'standing',
                                    'Class': cha_gpt.CLASS_NAMES[int(label)],
                                    'Person': 'D' if label == 2 else 'A' if label == 0 else 'B',
                                    'Session': f'{label}01.txt'}
                                   for label in np.repeat([0, 1, 2], 3)],
                'X_test': np.ones((6, 40, 2), dtype=np.float32), 'y_test': labels,
                'test_metadata': [{'Activity': activity, 'Class': cha_gpt.CLASS_NAMES[int(label)],
                                   'Person': 'D' if label == 2 else 'A' if label == 0 else 'B',
                                   'Session': f'{label}17.txt'}
                                  for activity, label in zip(
                                      ['standing', 'sitting', 'typing', 'standing', 'empty', 'empty'], labels)],
                'mean': np.zeros(2), 'std': np.ones(2), 'feature_dim': 2}

    def test_row_normalization_handles_unequal_support_and_missing_class(self):
        cm = np.array([[3, 1, 0], [2, 0, 0], [0, 0, 0]])
        normalized = cha_gpt.normalize_confusion_matrix(cm)
        np.testing.assert_allclose(normalized, [[0.75, 0.25, 0], [1, 0, 0], [0, 0, 0]])
        self.assertTrue(np.isfinite(normalized).all())

    def test_csv_axes_preserve_true_rows_predicted_columns(self):
        cm = np.array([[3, 1, 0], [2, 0, 0], [0, 0, 0]])
        with tempfile.TemporaryDirectory() as directory:
            cha_gpt.save_confusion_matrices(cm, 'Synthetic example', Path(directory, 'cm.png'))
            counts = pd.read_csv(Path(directory, 'cm_counts.csv'), index_col=0)
            normalized = pd.read_csv(Path(directory, 'cm_normalized.csv'), index_col=0)
            self.assertEqual(list(counts.index), ['Auth', 'Unauth', 'Empty'])
            self.assertEqual(counts.loc['Unauth', 'Auth'], 2)
            self.assertEqual(normalized.loc['Auth', 'Unauth'], 0.25)
            self.assertEqual(normalized.loc['Empty'].sum(), 0)
            self.assertTrue(Path(directory, 'cm.png').is_file())
            self.assertTrue(Path(directory, 'cm_normalized.png').is_file())

    def test_best_result_rejects_mismatched_protocol_and_uses_validation_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'A_cnn_best.json')
            path.write_text(json.dumps(self.best()))
            split = make_split(self.files(), 'A')
            with patch.object(evaluation, 'protocol_signature', return_value='matching-protocol'):
                _, epochs = evaluation.load_best_result(path, 'A', 'cnn', split)
                self.assertEqual(epochs, 2)
                _, epochs = evaluation.load_best_result(path, 'A', 'cnn', split, 25)
                self.assertEqual(epochs, 25)
                with self.assertRaises(ValueError):
                    evaluation.load_best_result(path, 'B', 'cnn', split)
            with patch.object(evaluation, 'protocol_signature', return_value='changed-protocol'):
                with self.assertRaises(ValueError):
                    evaluation.load_best_result(path, 'A', 'cnn', split)

    def test_final_refit_does_not_pass_final_test_to_fit_or_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            args = evaluation.parse_args(['--targets', 'A', '--models', 'cnn', '--output-dir', directory])
            data = self.data()
            model = SimpleNamespace(
                fit=Mock(return_value=SimpleNamespace(history={'loss': [1.0, 0.5], 'accuracy': [0.5, 1.0]})),
                predict=Mock(side_effect=lambda features, **kwargs: np.eye(3)[
                    data['y_train'] if features is data['X_train'] else [0, 1, 1, 2, 2, 0]]),
                save=Mock())
            with patch.object(evaluation, 'find_matching_files', side_effect=lambda person, _: self.files()[person]), \
                    patch.object(evaluation, 'validate_unique_recordings'), \
                    patch.object(evaluation, 'load_best_result', return_value=(self.best(), 2)), \
                    patch.object(evaluation, 'prepare_scenario_data', return_value=data) as prepare, \
                    patch.object(evaluation, 'recording_hashes', return_value=[]), \
                    patch.object(evaluation, 'build_tuning_model', return_value=model), \
                    patch.object(evaluation.tf.config, 'list_physical_devices', return_value=['fake-gpu']), \
                    patch.object(evaluation.tf.config.experimental, 'set_memory_growth'), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(args)
            split = prepare.call_args.args[0]
            self.assertEqual(sum(len(paths) for name, paths in split.items() if name.startswith('train_')), 48)
            self.assertEqual(sum(len(paths) for name, paths in split.items() if name.startswith('test_')), 12)
            self.assertIs(model.fit.call_args.args[0], data['X_train'])
            self.assertIs(model.fit.call_args.args[1], data['y_train'])
            self.assertNotIn('validation_data', model.fit.call_args.kwargs)
            self.assertNotIn('callbacks', model.fit.call_args.kwargs)
            self.assertEqual(model.fit.call_args.kwargs['epochs'], 2)
            self.assertIs(model.predict.call_args.args[0], data['X_test'])
            counts = pd.read_csv(Path(directory, 'cm_A_cnn_final_test_counts.csv'), index_col=0)
            np.testing.assert_array_equal(counts.values, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
            manifest = json.loads(Path(directory, 'evaluation_manifest.json').read_text())
            self.assertFalse(set(manifest[0]['train_files']) & set(manifest[0]['test_files']))
            train_metrics = pd.read_csv(Path(directory, 'A_cnn_train_session_metrics.csv'))
            self.assertEqual(train_metrics[train_metrics['Class'] == 'Empty']['Accuracy'].iloc[0], 1)
            test_metrics = pd.read_csv(Path(directory, 'A_cnn_test_session_metrics.csv'))
            self.assertEqual(test_metrics[test_metrics['Class'] == 'Empty']['Accuracy'].iloc[0], 0.5)
            probabilities = pd.read_csv(Path(directory, 'A_cnn_test_predictions.csv'))
            np.testing.assert_allclose(probabilities[['Probability_Auth', 'Probability_Unauth',
                                                     'Probability_Empty']].sum(axis=1), 1)
            model.save.assert_called_once_with(f'{directory}/A_cnn_model.keras')
            self.assertTrue(Path(directory, 'A_cnn_training_history.csv').is_file())

    def test_prepare_only_never_builds_predicts_or_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, 'not-created')
            args = evaluation.parse_args(['--targets', 'A', '--models', 'cnn', '--prepare-only',
                                          '--output-dir', str(output)])
            with patch.object(evaluation, 'find_matching_files', side_effect=lambda person, _: self.files()[person]), \
                    patch.object(evaluation, 'validate_unique_recordings'), \
                    patch.object(evaluation, 'load_best_result', return_value=(self.best(), 2)), \
                    patch.object(evaluation, 'prepare_scenario_data', return_value=self.data()), \
                    patch.object(evaluation, 'build_tuning_model') as build, \
                    patch.object(evaluation, 'save_confusion_matrices') as save, \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(args)
            build.assert_not_called()
            save.assert_not_called()
            self.assertFalse(output.exists())

    def test_diagnostic_predictions_preserve_shuffled_labels_and_confidence(self):
        labels = np.array([2, 0, 2, 1])
        metadata = [{'Person': 'D' if label == 2 else 'A' if label == 0 else 'B',
                     'Session': 'D01.txt' if label == 2 else f'{label}01.txt',
                     'Class': cha_gpt.CLASS_NAMES[int(label)]} for label in labels]
        probabilities = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1],
                                  [0.7, 0.1, 0.2], [0.1, 0.8, 0.1]])
        with tempfile.TemporaryDirectory() as directory:
            prediction, _, _ = evaluation.save_prediction_diagnostics(
                Path(directory, 'A_cnn'), 'train', labels, metadata, probabilities, 'A', '1D-CNN')
            np.testing.assert_array_equal(prediction, [2, 0, 0, 1])
            sessions = pd.read_csv(Path(directory, 'A_cnn_train_session_metrics.csv'))
            empty = sessions[sessions['Class'] == 'Empty'].iloc[0]
            self.assertEqual(empty['Windows'], 2)
            self.assertEqual(empty['Correct_Windows'], 1)
            self.assertEqual(empty['Accuracy'], 0.5)
            self.assertAlmostEqual(empty['Mean_Probability_Empty'], 0.45)

    def test_invalid_prediction_alignment_cannot_write_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory, 'A_cnn')
            metadata = [{'Person': 'D', 'Session': 'D01.txt', 'Class': 'Empty'}]
            for labels, probabilities in [(np.array([0]), np.array([[1, 0, 0]])),
                                          (np.array([2]), np.array([[0, 0, 0.4]])),
                                          (np.array([2]), np.array([[0, np.nan, 1]]))]:
                with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                    evaluation.save_prediction_diagnostics(prefix, 'train', labels, metadata,
                                                           probabilities, 'A', '1D-CNN')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_recording_manifest_changes_when_input_contents_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'D19.txt')
            path.write_text('old recording')
            first = evaluation.recording_hashes([str(path)])
            path.write_text('new recording')
            second = evaluation.recording_hashes([str(path)])
            self.assertEqual(first[0]['file'], second[0]['file'])
            self.assertNotEqual(first[0]['sha256'], second[0]['sha256'])

    def test_objective_exports_best_epoch_instead_of_stopping_epoch(self):
        data = self.data()
        data['X_validation'] = data.pop('X_test')
        data['y_validation'] = data.pop('y_test')
        attrs = {}
        trial = SimpleNamespace(set_user_attr=lambda key, value: attrs.update({key: value}))
        model = SimpleNamespace(
            fit=Mock(return_value=SimpleNamespace(history={'loss': [1, 0.8, 0.9, 1],
                                                          'val_macro_f1': [0.4, 0.8, 0.7, 0.6]})),
            predict=Mock(return_value=np.eye(3)[data['y_validation']]),
        )
        with patch.object(tuning, 'sample_parameters', return_value=self.best()['params']), \
                patch.object(tuning, 'prepare_tuning_data', return_value=data), \
                patch.object(tuning, 'build_tuning_model', return_value=model):
            objective, clear_cache = tuning.make_objective({}, 'cnn', 60)
            self.assertEqual(objective(trial), 1)
            clear_cache()
        self.assertEqual(attrs['best_epoch'], 2)
        self.assertEqual(attrs['epochs_run'], 4)


if __name__ == '__main__':
    unittest.main()
