import os
import argparse
import random
from pathlib import Path
from textwrap import fill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from csi_pipeline import (
    PREPROCESSING_VERSION, find_matching_files, make_split, prepare_scenario_data, validate_unique_recordings,
)

# TensorFlow 로그 최소화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# 0. 기본 설정
# ============================================================
SEED = 42
WINDOW_SEC = 2
STRIDE_SEC = 1
TARGET_FPS = 20
EPOCHS = 25
BATCH_SIZE = 32
SEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 타임스탬프 기준: 2초 윈도우 / 1초 간격 / 20 FPS
MAX_GAP_SEC = 0.5

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ['Auth', 'Unauth', 'Empty']
AUTH_TARGETS = ['A', 'B', 'C']


def configure_training_reproducibility(deterministic=True):
    """A random seed alone does not make GPU training reproducible."""
    tf.keras.utils.set_random_seed(SEED)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# ============================================================
# 3. 모델
# ============================================================
def build_1d_cnn(input_shape, num_classes=3, filters=64, kernel_size=5,
                 dropout=0.3, learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters, kernel_size=kernel_size, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters * 2, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_lstm(input_shape, num_classes=3, units=64, dropout=0.3,
               learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
        layers.Dropout(dropout),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ============================================================
# 5. 평가/저장
# ============================================================
def normalize_confusion_matrix(cm):
    """Normalize each true-label row; absent classes remain zero."""
    cm = np.asarray(cm)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals != 0)


def plot_confusion_matrix(cm, title, filename, normalize=False):
    cm = np.asarray(cm)
    if cm.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError('혼동행렬은 Auth / Unauth / Empty 순서의 3x3 배열이어야 합니다.')
    values = normalize_confusion_matrix(cm) if normalize else cm
    plt.figure(figsize=(7, 5.5))
    plt.imshow(values, interpolation='nearest', cmap='Blues',
               vmin=0, vmax=1 if normalize else None)
    plt.title(fill(title, width=55), fontsize=14, pad=12, fontweight='bold')
    colorbar = plt.colorbar()
    if normalize:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
    plt.yticks(tick_marks, CLASS_NAMES)

    threshold = 0.5 if normalize else values.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f'{values[i, j]:.1%}' if normalize else str(int(cm[i, j])),
                ha='center', va='center',
                color='white' if values[i, j] > threshold else 'black',
                fontweight='bold',
            )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_confusion_matrices(cm, title, filename):
    """Save counts and row proportions as PNG and labeled CSV."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, title, path)
    plot_confusion_matrix(cm, title + ' (row normalized)',
                          path.with_name(path.stem + '_normalized.png'), normalize=True)
    for suffix, values in [('counts', cm), ('normalized', normalize_confusion_matrix(cm))]:
        frame = pd.DataFrame(values, index=CLASS_NAMES, columns=CLASS_NAMES)
        frame.index.name = 'True / Predicted'
        frame.to_csv(path.with_name(path.stem + f'_{suffix}.csv'), encoding='utf-8-sig')


def evaluate_model(y_true, y_pred, scenario, model_name):
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average='macro',
        zero_division=0,
    )

    summary_row = {
        'Scenario': f'{scenario} Auth',
        'Model': model_name,
        'Accuracy': accuracy,
        'Macro_Precision': macro_p,
        'Macro_Recall': macro_r,
        'Macro_F1': macro_f1,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    class_rows = []
    for cls in CLASS_NAMES:
        class_rows.append({
            'Scenario': f'{scenario} Auth',
            'Model': model_name,
            'Class': cls,
            'Precision': report[cls]['precision'],
            'Recall': report[cls]['recall'],
            'F1': report[cls]['f1-score'],
            'Support': int(report[cls]['support']),
        })

    return summary_row, class_rows


def print_scenario_result(summary_row):
    print(
        f"  {summary_row['Model']:<7} | "
        f"Accuracy={summary_row['Accuracy']:.4f} | "
        f"Macro F1={summary_row['Macro_F1']:.4f} | "
        f"Macro Recall={summary_row['Macro_Recall']:.4f}"
    )


# ============================================================
# 6. 메인
# ============================================================
def evaluate_activities(y_true, y_pred, metadata, target, model_name):
    rows = []
    activities = np.asarray([row['Activity'] for row in metadata])
    for activity in ['standing', 'sitting', 'typing', 'empty']:
        mask = activities == activity
        if not mask.any():
            raise RuntimeError(f'{target}: {activity} 테스트 윈도우가 없습니다.')
        labels = [2] if activity == 'empty' else [0, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], labels=labels, average='macro', zero_division=0,
        )
        rows.append({
            'Scenario': f'{target} Auth',
            'Model': model_name,
            'Activity': activity,
            'Evaluated_Classes': '|'.join(CLASS_NAMES[label] for label in labels),
            'Accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'Macro_Precision': precision,
            'Macro_Recall': recall,
            'Macro_F1': f1,
            'Support': int(mask.sum()),
        })
    return rows


def main(prepare_only=False):
    files_all = {person: find_matching_files(person, SEARCH_DIR)
                 for person in ['A', 'B', 'C', 'D']}
    print('\n[파일 로드 상태]')
    for person, files in files_all.items():
        print(f'  {person}: {len(files)}개')
    splits = {target: make_split(files_all, target) for target in AUTH_TARGETS}
    print('\n[인가자별 실험 설계]')
    print('  Train: Auth 16 / Unauth 8+8 / Empty 16 = 48 files')
    print('  Test : Auth 4 / Unauth 2+2 / Empty 4 = 12 files')
    validate_unique_recordings(files_all)
    if not prepare_only:
        configure_training_reproducibility()

    summary_rows, class_rows, activity_rows = [], [], []
    prediction_rows, split_rows = [], []
    cm_total = {name: np.zeros((3, 3), dtype=int) for name in ['1D-CNN', 'LSTM']}
    for target, split in splits.items():
        print(f'\n========== [{target}가 인가자일 때] ==========')
        for name, files in split.items():
            print(f'  {name}: {len(files)} files')
            partition, role = name.split('_')
            for path in files:
                split_rows.append({
                    'Scenario': f'{target} Auth', 'Partition': partition,
                    'Class': {'auth': 'Auth', 'unauth': 'Unauth', 'empty': 'Empty'}[role],
                    'Person': os.path.basename(os.path.dirname(path)).upper(),
                    'Session': os.path.basename(path), 'Source_File': os.path.abspath(path),
                })
        data = prepare_scenario_data(
            split, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC,
            target_fps=TARGET_FPS, max_gap_sec=MAX_GAP_SEC, seed=SEED,
        )
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        print(f'  Input shape: {X_train.shape[1:]}')
        for partition in ['train', 'test']:
            counts = np.bincount(data[f'y_{partition}'], minlength=3)
            print(f'  {partition} windows: Auth={counts[0]}, Unauth={counts[1]}, Empty={counts[2]}')
        if prepare_only:
            continue

        weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=y_train)
        class_weight = {label: float(weight) for label, weight in enumerate(weights)}
        np.savez(f'csi_preprocessing_{target}.npz', mean=data['mean'], std=data['std'],
                 preprocessing_version=PREPROCESSING_VERSION,
                 feature_dim=data['feature_dim'], target_fps=TARGET_FPS,
                 window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, max_gap_sec=MAX_GAP_SEC)
        for model_name, builder, filename_tag in [
            ('1D-CNN', build_1d_cnn, '1d_cnn'), ('LSTM', build_lstm, 'lstm'),
        ]:
            print(f'[{target}] {model_name} 학습 시작')
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            model = builder(X_train.shape[1:])
            history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                shuffle=True, class_weight=class_weight, verbose=0)
            history_frame = pd.DataFrame(history.history)
            history_frame.insert(0, 'Epoch', range(1, len(history_frame) + 1))
            history_frame.to_csv(f'csi_training_history_{filename_tag}_{target}.csv',
                                 index=False, encoding='utf-8-sig')
            model.save(f'csi_model_{filename_tag}_{target}.keras')
            probabilities = model.predict(X_test, verbose=0)
            predictions = np.argmax(probabilities, axis=1)
            cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
            cm_total[model_name] += cm
            save_confusion_matrices(cm, f'{model_name} - {target} Auth',
                                  f'cm_{filename_tag}_{target}_auth.png')
            summary, details = evaluate_model(y_test, predictions, target, model_name)
            summary_rows.append(summary)
            class_rows.extend(details)
            activity_rows.extend(evaluate_activities(
                y_test, predictions, data['test_metadata'], target, model_name,
            ))
            for row, predicted, probability in zip(data['test_metadata'], predictions, probabilities):
                prediction_rows.append({
                    'Scenario': f'{target} Auth', 'Model': model_name, **row,
                    'Predicted_Class': CLASS_NAMES[int(predicted)],
                    'Probability_Auth': float(probability[0]),
                    'Probability_Unauth': float(probability[1]),
                    'Probability_Empty': float(probability[2]),
                })
            print_scenario_result(summary)
            del model
            tf.keras.backend.clear_session()

    if prepare_only:
        print('\n전처리 검증 완료. 학습 및 결과 파일 저장은 실행하지 않았습니다.')
        return

    df_summary, df_class = pd.DataFrame(summary_rows), pd.DataFrame(class_rows)
    metric_cols = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1']
    averages = df_summary.groupby('Model', as_index=False)[metric_cols].mean()
    averages.insert(0, 'Scenario', 'AVERAGE(A/B/C)')
    df_summary_final = pd.concat([df_summary, averages], ignore_index=True)
    df_summary_final[metric_cols] = df_summary_final[metric_cols].round(4)
    class_metric_cols = ['Precision', 'Recall', 'F1', 'Support']
    class_averages = df_class.groupby(['Model', 'Class'], as_index=False)[class_metric_cols].mean()
    class_averages.insert(0, 'Scenario', 'AVERAGE(A/B/C)')
    df_class_final = pd.concat([df_class, class_averages], ignore_index=True)
    df_class_final[class_metric_cols] = df_class_final[class_metric_cols].round(4)
    df_summary_final.to_csv('csi_3class_scenario_summary.csv', index=False, encoding='utf-8-sig')
    df_class_final.to_csv('csi_3class_class_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(activity_rows).to_csv('csi_activity_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(prediction_rows).to_csv('csi_test_window_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(split_rows).to_csv('csi_session_split.csv', index=False, encoding='utf-8-sig')
    for model_name, filename_tag in [('1D-CNN', '1d_cnn'), ('LSTM', 'lstm')]:
        save_confusion_matrices(cm_total[model_name], f'{model_name} - Aggregated A/B/C',
                              f'cm_{filename_tag}_aggregated.png')
    print('\n========== [최종 요약] ==========')
    print(df_summary_final.to_string(index=False))
    print('\n[저장] 성능 CSV 3개, 테스트 윈도우 예측 CSV, 세션 분할 CSV,')
    print('       정규화 설정 NPZ 3개, 혼동행렬 PNG 16개 및 혼동행렬 CSV 16개')
    print('       학습 기록 CSV 6개, 학습 모델 Keras 6개, 예측 CSV에 클래스별 확률 포함')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSI 인가자/비인가자/빈방 분류 실험')
    parser.add_argument('--prepare-only', action='store_true',
                        help='전처리만 확인하고 학습 및 결과 저장을 생략')
    main(prepare_only=parser.parse_args().prepare_only)
