"""Refit selected Optuna settings and evaluate sessions held out from tuning."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from cha_gpt import configure_training_reproducibility, evaluate_activities, evaluate_model, save_confusion_matrices
from csi_pipeline import (CLASS_NAMES, PREPROCESSING_VERSION, find_matching_files,
                          make_split, prepare_scenario_data,
                          select_experiment_recordings, validate_unique_recordings)
from tune_optuna import (
    MAX_GAP_SEC, SEARCH_DIR, SEARCH_SPACE, SEED, STRIDE_SEC, TARGET_FPS,
    build_tuning_model, make_tuning_split, protocol_signature, tf,
)

MODEL_NAMES = {'cnn': '1D-CNN', 'lstm': 'LSTM'}
ROOT = Path(__file__).resolve().parent


def recording_hashes(paths):
    records = []
    for path in paths:
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'file': str(path), 'sha256': digest.hexdigest()})
    return records


def save_prediction_diagnostics(prefix, partition, y_true, metadata, probabilities,
                                target, model_name):
    """Save per-window confidence and per-session accuracy for train or test."""
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    if (probabilities.shape != (len(y_true), len(CLASS_NAMES))
            or len(metadata) != len(y_true) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError('예측 확률/라벨/윈도우 행 수가 올바르지 않습니다.')
    if [row['Class'] for row in metadata] != [CLASS_NAMES[int(label)] for label in y_true]:
        raise ValueError('윈도우 메타데이터와 실제 라벨의 순서가 다릅니다.')
    predictions = np.argmax(probabilities, axis=1)
    frame = pd.DataFrame([
        {**row, 'Predicted_Class': CLASS_NAMES[int(label)],
         **{f'Probability_{name}': float(probability[index])
            for index, name in enumerate(CLASS_NAMES)}}
        for row, label, probability in zip(metadata, predictions, probabilities)
    ])
    frame.to_csv(f'{prefix}_{partition}_predictions.csv', index=False, encoding='utf-8-sig')
    sessions = []
    for (person, session, true_class), group in frame.groupby(['Person', 'Session', 'Class']):
        counts = group['Predicted_Class'].value_counts()
        sessions.append({
            'Person': person, 'Session': session, 'Class': true_class,
            'Windows': len(group), 'Correct_Windows': int(counts.get(true_class, 0)),
            'Accuracy': float((group['Class'] == group['Predicted_Class']).mean()),
            **{f'Predicted_{name}': int(counts.get(name, 0)) for name in CLASS_NAMES},
            **{f'Mean_Probability_{name}': float(group[f'Probability_{name}'].mean())
               for name in CLASS_NAMES},
        })
    pd.DataFrame(sessions).to_csv(f'{prefix}_{partition}_session_metrics.csv',
                                 index=False, encoding='utf-8-sig')
    summary, class_rows = evaluate_model(y_true, predictions, target, model_name)
    return predictions, summary, class_rows


def load_best_result(path, target, model_name, outer_split, epochs_override=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'최적화 결과가 없습니다: {path}. 먼저 tune_optuna.py를 실행하세요.')
    best = json.loads(path.read_text(encoding='utf-8'))
    if best.get('authorized_person') != target or best.get('model') != model_name:
        raise ValueError(f'{path}: 인가자 또는 모델이 요청 조건과 다릅니다.')
    if 'max_epochs' not in best:
        raise ValueError(f'{path}: 최적화 max_epochs 정보가 없습니다. 최신 코드로 결과를 저장하세요.')
    signature = protocol_signature(make_tuning_split(outer_split), model_name, best['max_epochs'],
                                   deterministic=best.get('training_determinism', False))
    if best.get('protocol_sha256') != signature:
        raise ValueError(f'{path}: 현재 학습 데이터 또는 최적화 설정과 결과의 실험 조건이 다릅니다.')
    params = best['params']
    for name in ('window_sec', 'batch_size', 'dropout'):
        if params.get(name) not in SEARCH_SPACE[name]:
            raise ValueError(f'{path}: 잘못된 {name} 설정입니다.')
    rate = params.get('learning_rate', 0)
    if not SEARCH_SPACE['learning_rate']['low'] <= rate <= SEARCH_SPACE['learning_rate']['high']:
        raise ValueError(f'{path}: 잘못된 learning_rate 설정입니다.')
    choices = {'filters': 'cnn_filters', 'kernel_size': 'cnn_kernel_size'} if model_name == 'cnn' else {'units': 'lstm_units'}
    for name, search_key in choices.items():
        if params.get(name) not in SEARCH_SPACE[search_key]:
            raise ValueError(f'{path}: 잘못된 {name} 설정입니다.')
    epochs = epochs_override if epochs_override is not None else best.get('diagnostics', {}).get('best_epoch')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(f'{path}: best_epoch가 없습니다. --epochs로 재학습 횟수를 지정하세요.')
    return best, epochs


def main(args):
    files = select_experiment_recordings(
        {person: find_matching_files(person, SEARCH_DIR) for person in 'ABCD'},
        args.sessions_per_person)
    validate_unique_recordings(files)
    # Validate all requested results before any fitting or output writes.
    jobs = []
    for target in args.targets:
        original_split = make_split(files, target, args.sessions_per_person)
        split = make_split(files, target, args.sessions_per_person,
                           split_seed=getattr(args, 'split_seed', None))
        for model_name in args.models:
            best, epochs = load_best_result(
                Path(args.best_dir) / f'{target}_{model_name}_best.json',
                target, model_name, original_split, args.epochs,
            )
            jobs.append((target, model_name, split, best, epochs))

    if not args.prepare_only:
        configure_training_reproducibility(args.deterministic)
        print(f'TensorFlow deterministic operations: {args.deterministic}')
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU가 없습니다. source scripts/activate_ubuntu.sh를 먼저 실행하세요.')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    summaries, details, activity_rows, manifest = [], [], [], []
    train_summaries, train_details = [], []
    totals = {name: np.zeros((3, 3), dtype=np.int64) for name in args.models}
    for target, model_name, split, best, epochs in jobs:
        params = best['params']
        if not args.prepare_only and best.get('training_determinism', False) != args.deterministic:
            print('기존 Optuna 설정을 재사용합니다. 최적화 때와 재학습의 결정론적 연산 설정이 다릅니다.')
        data = prepare_scenario_data(
            split, window_sec=params['window_sec'], stride_sec=STRIDE_SEC,
            target_fps=TARGET_FPS, max_gap_sec=MAX_GAP_SEC, seed=SEED,
        )
        train_n = sum(len(paths) for name, paths in split.items() if name.startswith('train_'))
        test_n = sum(len(paths) for name, paths in split.items() if name.startswith('test_'))
        print(f'{target} {model_name}: train {train_n} files {data["X_train"].shape}, '
              f'test {test_n} files {data["X_test"].shape}, epochs={epochs}')
        if args.prepare_only:
            continue
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prefix = output / f'{target}_{model_name}'
        train_files = [path for name, paths in split.items() if name.startswith('train_') for path in paths]
        test_files = [path for name, paths in split.items() if name.startswith('test_') for path in paths]
        train_recordings, test_recordings = recording_hashes(train_files), recording_hashes(test_files)
        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=data['y_train'])
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = None
        try:
            model = build_tuning_model(data['X_train'].shape[1:], model_name, params)
            # Epoch count comes from tuning validation (or the user's override).
            # Final test data is never passed to fit or a selection callback.
            history = model.fit(data['X_train'], data['y_train'], epochs=epochs,
                      batch_size=params['batch_size'], shuffle=True,
                      class_weight=dict(enumerate(weights)), verbose=2)
            history_frame = pd.DataFrame(history.history)
            history_frame.insert(0, 'Epoch', range(1, len(history_frame) + 1))
            history_frame.to_csv(f'{prefix}_training_history.csv', index=False, encoding='utf-8-sig')
            model.save(f'{prefix}_model.keras')
            train_probabilities = model.predict(data['X_train'], batch_size=params['batch_size'], verbose=0)
            _, train_summary, train_class_rows = save_prediction_diagnostics(
                prefix, 'train', data['y_train'], data['train_metadata'], train_probabilities,
                target, MODEL_NAMES[model_name])
            train_summaries.append(train_summary)
            train_details.extend(train_class_rows)
            test_probabilities = model.predict(data['X_test'], batch_size=params['batch_size'], verbose=0)
            predictions, summary, class_rows = save_prediction_diagnostics(
                prefix, 'test', data['y_test'], data['test_metadata'], test_probabilities,
                target, MODEL_NAMES[model_name])
            cm = confusion_matrix(data['y_test'], predictions, labels=[0, 1, 2])
            totals[model_name] += cm
            save_confusion_matrices(cm, f'{MODEL_NAMES[model_name]} - {target} Auth (final test)',
                                    output / f'cm_{target}_{model_name}_final_test.png')
            summaries.append(summary)
            details.extend(class_rows)
            activity_rows.extend(evaluate_activities(data['y_test'], predictions, data['test_metadata'],
                                                    target, MODEL_NAMES[model_name]))
            np.savez(f'{prefix}_preprocessing.npz', mean=data['mean'], std=data['std'],
                     preprocessing_version=PREPROCESSING_VERSION,
                     feature_dim=data['feature_dim'], window_sec=params['window_sec'],
                     target_fps=TARGET_FPS, stride_sec=STRIDE_SEC, max_gap_sec=MAX_GAP_SEC)
            manifest.append({'authorized_person': target, 'model': model_name, 'params': params,
                             'sessions_per_person': args.sessions_per_person,
                             'split_seed': getattr(args, 'split_seed', None),
                             'training_seed': SEED,
                             'hyperparameters_reoptimized': False,
                             'parameter_selection_test_overlap': sorted(
                                 set(test_files) & {p for name, paths in
                                     make_tuning_split(make_split(files, target, args.sessions_per_person)).items()
                                     for p in paths}),
                             'training_determinism': args.deterministic,
                             'optimization_training_determinism': best.get('training_determinism', False),
                             'epochs': epochs, 'best_trial': best['best_trial'],
                             'protocol_sha256': best['protocol_sha256'],
                             'train_files': train_files, 'test_files': test_files,
                             'train_recordings': train_recordings, 'test_recordings': test_recordings,
                             'preprocessing_version': PREPROCESSING_VERSION,
                             'model_file': f'{prefix.name}_model.keras'})
            train_empty = next(row for row in train_class_rows if row['Class'] == 'Empty')
            test_empty = next(row for row in class_rows if row['Class'] == 'Empty')
            print(f'{target} {model_name}: Empty recall train={train_empty["Recall"]:.4f} / '
                  f'test={test_empty["Recall"]:.4f}')
            print(f'{target} {model_name}: final test Macro F1={summary["Macro_F1"]:.4f}')
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()

    if args.prepare_only:
        print('전처리 검증 완료. 재학습, 테스트 예측 및 결과 저장은 실행하지 않았습니다.')
        return
    output = Path(args.output_dir)
    for model_name, cm in totals.items():
        if len(args.targets) > 1:
            save_confusion_matrices(cm, f'{MODEL_NAMES[model_name]} - Combined final tests ({"/".join(args.targets)})',
                                    output / f'cm_{model_name}_combined_final_test.png')
    for name, rows in [('summary', summaries), ('class_metrics', details), ('activity_metrics', activity_rows)]:
        pd.DataFrame(rows).to_csv(output / f'final_test_{name}.csv', index=False, encoding='utf-8-sig')
    for name, rows in [('summary', train_summaries), ('class_metrics', train_details)]:
        pd.DataFrame(rows).to_csv(output / f'final_train_{name}.csv', index=False, encoding='utf-8-sig')
    (output / 'evaluation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'혼동행렬 및 최종 테스트 평가 저장: {output.resolve()}')
    print('학습/테스트 파일별 정확도, 클래스별 확률, 학습 기록, 모델 및 입력 파일 해시도 저장했습니다.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Optuna 최적 설정 재학습 및 최종 테스트 혼동행렬')
    parser.add_argument('--targets', nargs='+', choices=list('ABC'), default=list('ABC'))
    parser.add_argument('--models', nargs='+', choices=['cnn', 'lstm'], default=['cnn', 'lstm'])
    parser.add_argument('--best-dir', default=str(ROOT / 'optuna_results'))
    parser.add_argument('--output-dir', default=str(ROOT / 'optuna_evaluation'))
    parser.add_argument('--epochs', type=int, help='최적 Trial의 best_epoch 대신 사용할 재학습 epoch 수')
    parser.add_argument('--sessions-per-person', type=int, choices=[15, 20], default=20)
    parser.add_argument('--split-seed', type=int, help='원본 파일 단위 무작위 분할 seed; 생략하면 번호순 분할')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error('epochs는 1 이상이어야 합니다.')
    if args.split_seed is not None and args.split_seed < 0:
        parser.error('split-seed는 0 이상이어야 합니다.')
    args.targets = list(dict.fromkeys(args.targets))
    args.models = list(dict.fromkeys(args.models))
    return args


if __name__ == '__main__':
    main(parse_args())
