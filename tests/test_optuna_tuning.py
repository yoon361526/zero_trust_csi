import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna

import tune_optuna as tuning
from csi_pipeline import make_split


class TuningTests(unittest.TestCase):
    def recordings(self):
        return {person: [f'/data/{person}/{person}{i:02d}.txt' for i in range(1, 21)]
                for person in 'ABCD'}

    def test_holdout_uses_only_outer_training_files_for_all_people(self):
        for target in 'ABC':
            outer = make_split(self.recordings(), target)
            inner = tuning.make_tuning_split(outer)
            self.assertEqual([len(inner[name]) for name in
                              ['train_auth', 'train_unauth', 'train_empty',
                               'test_auth', 'test_unauth', 'test_empty']],
                             [12, 12, 12, 4, 4, 4])
            fit = {p for name, paths in inner.items() if name.startswith('train_') for p in paths}
            validation = {p for name, paths in inner.items() if name.startswith('test_') for p in paths}
            outer_train = {p for name, paths in outer.items() if name.startswith('train_') for p in paths}
            outer_test = {p for name, paths in outer.items() if name.startswith('test_') for p in paths}
            self.assertEqual(fit | validation, outer_train)
            self.assertFalse(fit & validation)
            self.assertFalse((fit | validation) & outer_test)
            self.assertEqual(inner['test_auth'], outer['train_auth'][12:16])

    def test_preprocessor_receives_validation_instead_of_final_test(self):
        outer = make_split(self.recordings(), 'A')
        inner = tuning.make_tuning_split(outer)
        data = {'X_test': 'validation features', 'y_test': 'validation labels',
                'test_metadata': ['validation row']}
        with patch.object(tuning, 'prepare_scenario_data', return_value=data) as prepare:
            result = tuning.prepare_tuning_data(inner, 4)
        self.assertEqual(prepare.call_args.args[0], inner)
        self.assertEqual(result['X_validation'], 'validation features')
        self.assertNotIn('X_test', result)

    def test_reduced_holdout_is_balanced_and_excludes_final_test_for_all_targets(self):
        for target in 'ABC':
            outer = make_split(self.recordings(), target, sessions_per_person=15)
            inner = tuning.make_tuning_split(outer)
            self.assertEqual([len(inner[name]) for name in
                              ['train_auth', 'train_unauth', 'train_empty',
                               'test_auth', 'test_unauth', 'test_empty']],
                             [9, 9, 9, 3, 3, 3])
            fit = {p for name, paths in inner.items() if name.startswith('train_') for p in paths}
            validation = {p for name, paths in inner.items() if name.startswith('test_') for p in paths}
            outer_train = {p for name, paths in outer.items() if name.startswith('train_') for p in paths}
            outer_test = {p for name, paths in outer.items() if name.startswith('test_') for p in paths}
            self.assertEqual(fit | validation, outer_train)
            self.assertFalse(fit & validation)
            self.assertFalse((fit | validation) & outer_test)

    def test_reduced_prepare_only_passes_selected_recordings_without_creating_studies(self):
        args = tuning.parse_args(['--sessions-per-person', '15', '--targets', 'A', '--prepare-only'])
        files = self.recordings()
        data = {'X_train': np.zeros((3, 40, 1)), 'X_validation': np.zeros((3, 40, 1))}
        with patch.object(tuning, 'find_matching_files', side_effect=lambda key, directory: files[key]), \
                patch.object(tuning, 'validate_unique_recordings') as unique, \
                patch.object(tuning, 'prepare_tuning_data', return_value=data), \
                patch.object(tuning.optuna, 'create_study') as create, \
                patch.object(tuning.Path, 'mkdir') as mkdir, \
                contextlib.redirect_stdout(io.StringIO()):
            tuning.main(args)
        self.assertEqual([len(paths) for paths in unique.call_args.args[0].values()], [15] * 4)
        create.assert_not_called()
        mkdir.assert_not_called()

    def test_search_parameters_select_only_relevant_model_options(self):
        common = {'window_sec': 2, 'batch_size': 32, 'dropout': 0.3, 'learning_rate': 0.001}
        cnn = tuning.sample_parameters(optuna.trial.FixedTrial({**common, 'filters': 64, 'kernel_size': 5}), 'cnn')
        lstm = tuning.sample_parameters(optuna.trial.FixedTrial({**common, 'units': 64}), 'lstm')
        self.assertEqual(cnn, {**common, 'filters': 64, 'kernel_size': 5})
        self.assertEqual(lstm, {**common, 'units': 64})

    def test_macro_f1_callback_reports_and_prunes_without_training(self):
        trial = SimpleNamespace(report=lambda score, step: reported.append((score, step)),
                                should_prune=lambda: False)
        reported = []
        callback = tuning.ValidationMacroF1(trial, np.zeros((3, 40, 1)), np.array([0, 1, 2]), 32)
        callback.set_model(SimpleNamespace(predict=lambda *args, **kwargs: np.eye(3)))
        logs = {}
        callback.on_epoch_end(0, logs)
        self.assertEqual(logs['val_macro_f1'], 1)
        self.assertEqual(reported, [(1, 0)])
        trial.should_prune = lambda: True
        with self.assertRaises(optuna.TrialPruned):
            callback.on_epoch_end(1, {})

    def test_prepare_only_does_not_create_studies_or_output_directories(self):
        args = tuning.parse_args(['--targets', 'A', '--prepare-only'])
        data = {'X_train': np.zeros((3, 40, 1)), 'X_validation': np.zeros((3, 40, 1))}
        files = self.recordings()
        with patch.object(tuning, 'find_matching_files', side_effect=lambda key, directory: files[key]), \
                patch.object(tuning, 'validate_unique_recordings'), \
                patch.object(tuning, 'prepare_tuning_data', return_value=data), \
                patch.object(tuning.optuna, 'create_study') as create, \
                patch.object(tuning.Path, 'mkdir') as mkdir, \
                contextlib.redirect_stdout(io.StringIO()):
            tuning.main(args)
            create.assert_not_called()
            mkdir.assert_not_called()

    def test_changed_data_or_settings_cannot_resume_a_study(self):
        study = SimpleNamespace(user_attrs={'protocol_sha256': 'old'}, trials=[],
                                set_user_attr=lambda key, value: None)
        with self.assertRaises(ValueError):
            tuning.require_matching_protocol(study, 'new')
        study.user_attrs = {}
        study.trials = ['unknown old trial']
        with self.assertRaises(ValueError):
            tuning.require_matching_protocol(study, 'new')

    def test_protocol_signature_tracks_file_contents_and_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'A01.txt'
            path.write_text('first recording')
            split = {'train_auth': [str(path)]}
            first = tuning.protocol_signature(split, 'cnn', 60)
            self.assertNotEqual(first, tuning.protocol_signature(split, 'cnn', 30))
            path.write_text('corrected recording')
            self.assertNotEqual(first, tuning.protocol_signature(split, 'cnn', 60))

    def test_deterministic_mode_cannot_resume_non_deterministic_study(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'A01.txt')
            path.write_text('same recording')
            split = {'train_auth': [str(path)]}
            legacy = tuning.protocol_signature(split, 'cnn', 60)
            self.assertEqual(legacy, tuning.protocol_signature(split, 'cnn', 60, deterministic=False))
            corrected = tuning.protocol_signature(split, 'cnn', 60, deterministic=True)
            self.assertNotEqual(legacy, corrected)
            study = SimpleNamespace(user_attrs={'protocol_sha256': legacy}, trials=[],
                                    set_user_attr=lambda key, value: None)
            with self.assertRaises(ValueError):
                tuning.require_matching_protocol(study, corrected)

    def test_training_config_enables_determinism_and_seeds_all_generators(self):
        import cha_gpt
        with patch.object(cha_gpt.tf.keras.utils, 'set_random_seed') as seed, \
                patch.object(cha_gpt.tf.config.experimental, 'enable_op_determinism') as enable:
            cha_gpt.configure_training_reproducibility()
        seed.assert_called_once_with(42)
        enable.assert_called_once()
        self.assertTrue(tuning.parse_args([]).deterministic)

    def test_preprocessing_change_cannot_resume_old_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'A01.txt')
            path.write_text('unchanged recording')
            split = {'train_auth': [str(path)]}
            with patch.object(tuning, 'PREPROCESSING_VERSION', 1):
                old_signature = tuning.protocol_signature(split, 'cnn', 60)
            new_signature = tuning.protocol_signature(split, 'cnn', 60)
            self.assertNotEqual(old_signature, new_signature)
            study = SimpleNamespace(user_attrs={'protocol_sha256': old_signature}, trials=[],
                                    set_user_attr=lambda key, value: None)
            with self.assertRaises(ValueError):
                tuning.require_matching_protocol(study, new_signature)

    def test_sqlite_resume_and_result_export_without_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = f'sqlite:///{Path(directory, "study.sqlite3").as_posix()}'
            study = optuna.create_study(study_name='test', storage=storage, direction='maximize')
            tuning.require_matching_protocol(study, 'test-protocol')
            study.add_trial(optuna.trial.create_trial(
                value=0.75, params={'dropout': 0.3},
                distributions={'dropout': optuna.distributions.CategoricalDistribution([0.2, 0.3, 0.5])},
            ))
            resumed = optuna.create_study(study_name='test', storage=storage,
                                          direction='maximize', load_if_exists=True)
            tuning.require_matching_protocol(resumed, 'test-protocol')
            self.assertEqual(resumed.best_value, 0.75)
            tuning.export_results(resumed, 'A', 'cnn', Path(directory))
            import json
            result = json.loads(Path(directory, 'A_cnn_best.json').read_text())
            self.assertFalse(result['final_test_used'])
            self.assertEqual(result['params'], {'dropout': 0.3})
            self.assertTrue(Path(directory, 'A_cnn_trials.csv').is_file())

    def test_parameterized_builders_preserve_shapes_and_apply_settings(self):
        with tuning.tf.device('/CPU:0'):
            cnn = tuning.build_tuning_model((40, 192), 'cnn',
                                            {'filters': 32, 'kernel_size': 3,
                                             'dropout': 0.5, 'learning_rate': 0.0001})
            self.assertEqual(cnn.output_shape, (None, 3))
            self.assertEqual(cnn.layers[0].filters, 32)
            self.assertEqual(cnn.layers[3].filters, 64)
            self.assertAlmostEqual(float(cnn.optimizer.learning_rate.numpy()), 0.0001)
            lstm = tuning.build_tuning_model((40, 192), 'lstm',
                                             {'units': 32, 'dropout': 0.5, 'learning_rate': 0.0003})
            self.assertEqual(lstm.output_shape, (None, 3))
            self.assertEqual(lstm.layers[0].forward_layer.units, 32)
            self.assertEqual(lstm.layers[2].units, 32)
        tuning.tf.keras.backend.clear_session()

    def test_validation_callback_and_early_stopping_restore_best_epoch(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        epoch_predictions = [[0, 1, 1, 2, 2, 0], labels.tolist(), [0, 0, 1, 1, 0, 0]]
        reports = []
        trial = SimpleNamespace(report=lambda value, step: reports.append((value, step)),
                                should_prune=lambda: False)

        class MarkerModel:
            marker = 0
            stop_training = False

            def predict(self, values, **kwargs):
                return np.eye(3)[epoch_predictions[self.marker]]

            def get_weights(self):
                return [np.array([self.marker])]

            def set_weights(self, weights):
                self.marker = int(weights[0][0])

        model = MarkerModel()
        scorer = tuning.ValidationMacroF1(trial, np.zeros((6, 40, 1)), labels, 16)
        stopping = tuning.tf.keras.callbacks.EarlyStopping(
            monitor='val_macro_f1', mode='max', patience=1, restore_best_weights=True)
        for callback in (scorer, stopping):
            callback.set_model(model)
            callback.on_train_begin()
        for epoch in range(3):
            model.marker = epoch
            logs = {}
            scorer.on_epoch_end(epoch, logs)
            stopping.on_epoch_end(epoch, logs)
        stopping.on_train_end()
        self.assertTrue(model.stop_training)
        self.assertEqual(model.marker, 1)
        self.assertEqual(reports[1], (1.0, 1))
        self.assertLess(reports[2][0], reports[1][0])

if __name__ == '__main__':
    unittest.main()
