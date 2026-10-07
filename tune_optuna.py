"""Optuna tuning with a fixed session holdout; final test files are never scored."""

import argparse
import gc
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from cha_gpt import (
    MAX_GAP_SEC, SEARCH_DIR, SEED, STRIDE_SEC, TARGET_FPS,
    build_1d_cnn, build_lstm, configure_training_reproducibility, tf,
)
from csi_pipeline import (
    PREPROCESSING_VERSION, REDUCED_TEST_COUNTS, find_matching_files, make_split,
    prepare_scenario_data, select_experiment_recordings, validate_unique_recordings,
)

SEARCH_SPACE = {
    'window_sec': [1, 2, 4],
    'batch_size': [16, 32, 64],
    'dropout': [0.2, 0.3, 0.5],
    'learning_rate': {'low': 1e-4, 'high': 1e-3, 'log': True},
    'cnn_filters': [32, 64],
    'cnn_kernel_size': [3, 5, 7],
    'lstm_units': [32, 64, 128],
}


def make_tuning_split(outer_split):
    """A fixed 75/25 holdout drawn exclusively from outer training sessions."""
    others = {}
    for path in outer_split['train_unauth']:
        others.setdefault(Path(path).parent.name, []).append(path)
    count = len(outer_split['train_auth'])
    if (count not in (12, 16) or len(outer_split['train_empty']) != count
            or len(others) != 2 or any(len(paths) != count // 2 for paths in others.values())):
        raise ValueError('최적화 입력은 Auth/Empty 12 또는 16, Unauth는 사람별 절반이어야 합니다.')
    fit_count = count * 3 // 4
    target = Path(outer_split['train_auth'][0]).parent.name
    validation_counts = (REDUCED_TEST_COUNTS[target] if count == 12
                          else {person: 2 for person in others})
    split = {
        'train_auth': outer_split['train_auth'][:fit_count],
        'test_auth': outer_split['train_auth'][fit_count:],
        'train_unauth': [p for person, paths in others.items()
                         for p in paths[:-validation_counts[person]]],
        'test_unauth': [p for person, paths in others.items()
                        for p in paths[-validation_counts[person]:]],
        'train_empty': outer_split['train_empty'][:fit_count],
        'test_empty': outer_split['train_empty'][fit_count:],
    }
    # The shared preprocessor uses test_* keys for the held-out partition.
    # In this dictionary those keys mean validation, not the final test set.
    fit_files = {p for name, paths in split.items() if name.startswith('train_') for p in paths}
    validation_files = {p for name, paths in split.items() if name.startswith('test_') for p in paths}
    final_test = {p for name, paths in outer_split.items() if name.startswith('test_') for p in paths}
    if fit_files & validation_files or (fit_files | validation_files) & final_test:
        raise ValueError('학습/검증/최종 테스트 파일 분할이 겹칩니다.')
    return split


def prepare_tuning_data(split, window_sec):
    data = prepare_scenario_data(
        split, window_sec=window_sec, stride_sec=STRIDE_SEC,
        target_fps=TARGET_FPS, max_gap_sec=MAX_GAP_SEC, seed=SEED,
    )
    data['X_validation'] = data.pop('X_test')
    data['y_validation'] = data.pop('y_test')
    data['validation_metadata'] = data.pop('test_metadata')
    return data


def sample_parameters(trial, model_name):
    params = {name: trial.suggest_categorical(name, SEARCH_SPACE[name])
              for name in ['window_sec', 'batch_size', 'dropout']}
    params['learning_rate'] = trial.suggest_float('learning_rate', **SEARCH_SPACE['learning_rate'])
    if model_name == 'cnn':
        params['filters'] = trial.suggest_categorical('filters', SEARCH_SPACE['cnn_filters'])
        params['kernel_size'] = trial.suggest_categorical('kernel_size', SEARCH_SPACE['cnn_kernel_size'])
    elif model_name == 'lstm':
        params['units'] = trial.suggest_categorical('units', SEARCH_SPACE['lstm_units'])
    else:
        raise ValueError(f'알 수 없는 모델: {model_name}')
    return params


def build_tuning_model(input_shape, model_name, params):
    kwargs = {name: params[name] for name in ['dropout', 'learning_rate']}
    if model_name == 'cnn':
        return build_1d_cnn(input_shape, filters=params['filters'],
                            kernel_size=params['kernel_size'], **kwargs)
    return build_lstm(input_shape, units=params['units'], **kwargs)


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, trial, X_validation, y_validation, batch_size):
        super().__init__()
        self.trial = trial
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        predictions = np.argmax(self.model.predict(
            self.X_validation, batch_size=self.batch_size, verbose=0,
        ), axis=1)
        score = f1_score(self.y_validation, predictions, labels=[0, 1, 2],
                         average='macro', zero_division=0)
        if logs is not None:
            logs['val_macro_f1'] = float(score)
        self.trial.report(float(score), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f'epoch {epoch + 1}: validation Macro F1={score:.4f}')


def make_objective(split, model_name, max_epochs):
    @lru_cache(maxsize=3)
    def data_for_window(window_sec):
        return prepare_tuning_data(split, window_sec)

    def objective(trial):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        params = sample_parameters(trial, model_name)
        data = data_for_window(params['window_sec'])
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]),
                                       y=data['y_train'])
        class_weight = {label: float(weight) for label, weight in enumerate(weights)}
        model = None
        try:
            model = build_tuning_model(data['X_train'].shape[1:], model_name, params)
            score_callback = ValidationMacroF1(
                trial, data['X_validation'], data['y_validation'], params['batch_size'],
            )
            stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_macro_f1', mode='max', patience=5, restore_best_weights=True,
            )
            history = model.fit(
                data['X_train'], data['y_train'], epochs=max_epochs,
                batch_size=params['batch_size'], shuffle=True, class_weight=class_weight,
                callbacks=[score_callback, stopping], verbose=0,
            )
            predictions = np.argmax(model.predict(
                data['X_validation'], batch_size=params['batch_size'], verbose=0,
            ), axis=1)
            score = f1_score(data['y_validation'], predictions, labels=[0, 1, 2],
                             average='macro', zero_division=0)
            negative = data['y_validation'] != 0
            trial.set_user_attr('epochs_run', len(history.history['loss']))
            trial.set_user_attr('best_epoch', int(np.argmax(history.history['val_macro_f1'])) + 1)
            trial.set_user_attr('validation_accuracy', float(accuracy_score(data['y_validation'], predictions)))
            trial.set_user_attr('false_auth_rate', float(np.mean(predictions[negative] == 0)))
            return float(score)
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    return objective, data_for_window.cache_clear


def protocol_signature(split, model_name, max_epochs, deterministic=False):
    files = {}
    for group, paths in split.items():
        files[group] = []
        for path in paths:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            files[group].append({'session': f'{Path(path).parent.name}/{Path(path).name}',
                                 'sha256': digest.hexdigest()})
    protocol = {'version': 1, 'preprocessing_version': PREPROCESSING_VERSION,
                'model': model_name, 'files': files,
                'max_epochs': max_epochs, 'seed': SEED, 'fps': TARGET_FPS,
                'stride_sec': STRIDE_SEC, 'max_gap_sec': MAX_GAP_SEC,
                'search_space': SEARCH_SPACE, 'tensorflow': tf.__version__,
                'optuna': optuna.__version__}
    # Preserve old signatures only for explicitly non-deterministic results.
    if deterministic:
        protocol['deterministic_ops'] = True
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def require_matching_protocol(study, signature):
    existing = study.user_attrs.get('protocol_sha256')
    if existing is None and study.trials:
        raise ValueError('기존 Study의 실험 조건을 확인할 수 없습니다. 새 저장 폴더를 사용하세요.')
    if existing is not None and existing != signature:
        raise ValueError('데이터 또는 탐색 설정이 기존 Study와 다릅니다. 새 저장 폴더를 사용하세요.')
    study.set_user_attr('protocol_sha256', signature)


def export_results(study, target, model_name, output_dir, max_epochs=60, deterministic=False):
    prefix = output_dir / f'{target}_{model_name}'
    study.trials_dataframe().to_csv(f'{prefix}_trials.csv', index=False, encoding='utf-8-sig')
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        raise RuntimeError('완료된 Trial이 없습니다. 실패/가지치기 기록을 확인하세요.')
    best = study.best_trial
    payload = {'authorized_person': target, 'model': model_name,
               'best_trial': best.number, 'validation_macro_f1': best.value,
               'params': best.params, 'diagnostics': best.user_attrs,
               'protocol_sha256': study.user_attrs['protocol_sha256'],
               'max_epochs': max_epochs,
               'training_determinism': deterministic,
               'final_test_used': False}
    Path(f'{prefix}_best.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def main(args):
    available = {person: find_matching_files(person, SEARCH_DIR) for person in 'ABCD'}
    files_all = select_experiment_recordings(available, args.sessions_per_person)
    outer_splits = {target: make_split(files_all, target, args.sessions_per_person) for target in args.targets}
    validate_unique_recordings(files_all)
    splits = {target: make_tuning_split(split) for target, split in outer_splits.items()}
    for target, split in splits.items():
        fit_n = sum(len(paths) for name, paths in split.items() if name.startswith('train_'))
        val_n = sum(len(paths) for name, paths in split.items() if name.startswith('test_'))
        test_n = sum(len(paths) for name, paths in outer_splits[target].items() if name.startswith('test_'))
        print(f'{target}: tuning train {fit_n} / validation {val_n} / reserved final test {test_n}')
    if args.prepare_only:
        for target, split in splits.items():
            data = prepare_tuning_data(split, window_sec=2)
            print(f"{target}: train={data['X_train'].shape}, validation={data['X_validation'].shape}")
        print('전처리 검증 완료. Optuna Trial과 학습, 결과 저장은 실행하지 않았습니다.')
        return

    configure_training_reproducibility(args.deterministic)
    print(f'TensorFlow deterministic operations: {args.deterministic}')
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('GPU가 없습니다. Ubuntu에서 source scripts/activate_ubuntu.sh를 먼저 실행하세요.')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    storage_dir = Path(args.storage_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        'sessions_per_person': args.sessions_per_person,
        'selection_rule': 'first numbered sessions; no selection based on model accuracy',
        'selected': files_all,
        'excluded': {person: [path for path in paths if path not in files_all[person]]
                     for person, paths in available.items()},
        'outer_splits': outer_splits, 'tuning_splits': splits,
        'seed': SEED, 'folds': False, 'training_determinism': args.deterministic,
    }
    (output_dir / 'selection_manifest.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(splits) * len(args.models)} Studies, each adding {args.trials} Trials')
    for target, split in splits.items():
        for model_name in args.models:
            signature = protocol_signature(split, model_name, args.max_epochs, args.deterministic)
            database = storage_dir / f'{target}_{model_name}.sqlite3'
            study = optuna.create_study(
                study_name=f'{target}_{model_name}', storage=f'sqlite:///{database.as_posix()}',
                direction='maximize', load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            )
            require_matching_protocol(study, signature)
            objective, clear_cache = make_objective(split, model_name, args.max_epochs)
            try:
                study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True,
                               catch=(tf.errors.ResourceExhaustedError,))
                export_results(study, target, model_name, output_dir, args.max_epochs, args.deterministic)
            finally:
                clear_cache()
                gc.collect()
            print(f'{target} {model_name}: best validation Macro F1={study.best_value:.4f}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='파일 단위 고정 검증 세트를 사용하는 Optuna 최적화')
    parser.add_argument('--targets', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    parser.add_argument('--models', nargs='+', choices=['cnn', 'lstm'], default=['cnn', 'lstm'])
    parser.add_argument('--trials', type=int, default=20, help='각 Study에 추가할 Trial 수')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--sessions-per-person', type=int, choices=[15, 20], default=20,
                        help='15는 대상별 01~15만 사용하며 원본은 삭제하지 않습니다.')
    parser.add_argument('--storage-dir', default=str(Path.home() / '.local/share/zero_trust/optuna'))
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'optuna_results'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                        help='기본값은 재현 가능한 연산. 이전 실행 비교에만 --no-deterministic을 사용합니다.')
    args = parser.parse_args(argv)
    if args.trials < 1 or args.max_epochs < 1:
        parser.error('trials와 max-epochs는 1 이상이어야 합니다.')
    args.targets = list(dict.fromkeys(args.targets))
    args.models = list(dict.fromkeys(args.models))
    return args


if __name__ == '__main__':
    main(parse_args())
