import os
import re
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

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

# 현재 중간점검 설계
# A/B/C: 각 20세션 -> Train 16 / Test 4
# D    : 4세션      -> Train 3 / Test 1
ABC_TRAIN_N = 16
ABC_TEST_N = 4
D_TRAIN_N = 3
D_TEST_N = 1

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ['Auth', 'Unauth', 'Empty']
SCENARIOS = ['A', 'B', 'C']


# ============================================================
# 1. 파일 검색
# ============================================================
def find_matching_files(prefix, search_dir=SEARCH_DIR):
    """
    data/A, data/B, data/C, data/D 중 해당 폴더에서 세션 파일을 검색.
    파일명 안의 A01, A02 ... / B01 ... / C01 ... / D01 ... 형태를 인식.
    예: A01.txt, csi_A01_20261005.txt
    """
    session_dir = os.path.join(search_dir, prefix)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f'데이터 폴더를 찾을 수 없습니다: {session_dir}')

    candidates = glob.glob(os.path.join(session_dir, '*.txt'))
    found = []

    pattern = re.compile(
        rf'(?i)(?:^|[^A-Za-z0-9]){re.escape(prefix)}0*(\d+)(?:[^0-9]|$)'
    )

    for path in candidates:
        name = os.path.basename(path)
        m = pattern.search(name)
        if m:
            session_no = int(m.group(1))
            found.append((session_no, path))

    found.sort(key=lambda x: (x[0], x[1]))

    # 같은 세션 번호가 여러 개 잡히면 첫 번째 파일만 사용
    unique = {}
    for session_no, path in found:
        unique.setdefault(session_no, path)

    return [unique[k] for k in sorted(unique)]


# ============================================================
# 2. CSI 파싱 + Sliding Window
# ============================================================
def read_csi_amplitude(filepath):
    records = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'CSI_DATA' not in line:
                continue

            m = re.search(r'"\[(.*?)\]"', line) or re.search(r'\[(.*?)\]', line)
            if not m:
                continue

            try:
                vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
                if vals:
                    records.append(vals)
            except ValueError:
                continue

    if not records:
        return None

    # 파일 내부에서 가장 자주 등장하는 CSI 길이 사용
    lengths, counts = np.unique([len(r) for r in records], return_counts=True)
    target_len = int(lengths[np.argmax(counts)])
    valid_records = [r for r in records if len(r) == target_len]

    if not valid_records:
        return None

    csi_matrix = np.asarray(valid_records, dtype=np.float32)

    # I/Q pair 보장
    if csi_matrix.shape[1] % 2 != 0:
        csi_matrix = csi_matrix[:, :-1]

    I = csi_matrix[:, 0::2]
    Q = csi_matrix[:, 1::2]
    amp = np.sqrt(I ** 2 + Q ** 2)

    return amp.astype(np.float32)


def determine_global_feature_dim(file_groups):
    """
    모든 사용 파일을 한 번 훑어 가장 흔한 amplitude feature dimension을 선택.
    파일별 차원 불일치로 np.array가 깨지는 것을 방지.
    """
    dims = []
    checked = set()

    for group in file_groups:
        for filepath in group:
            if filepath in checked:
                continue
            checked.add(filepath)

            amp = read_csi_amplitude(filepath)
            if amp is not None:
                dims.append(amp.shape[1])

    if not dims:
        raise RuntimeError('사용 가능한 CSI 파일을 읽지 못했습니다.')

    values, counts = np.unique(dims, return_counts=True)
    feature_dim = int(values[np.argmax(counts)])
    print(f'[CSI] 공통 feature dimension = {feature_dim}')
    return feature_dim


def parse_csi_to_windows(
    file_list,
    expected_feature_dim,
    window_sec=WINDOW_SEC,
    stride_sec=STRIDE_SEC,
    target_fps=TARGET_FPS,
):
    window_size = int(window_sec * target_fps)
    stride_size = int(stride_sec * target_fps)
    windows = []

    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f'[경고] 파일 없음: {filepath}')
            continue

        amp = read_csi_amplitude(filepath)
        if amp is None:
            print(f'[경고] CSI_DATA를 읽지 못함: {filepath}')
            continue

        if amp.shape[1] != expected_feature_dim:
            print(
                f'[경고] feature dimension 불일치로 제외: {filepath} '
                f'({amp.shape[1]} != {expected_feature_dim})'
            )
            continue

        # 기존 기획을 유지한 세션 단위 Z-score 정규화
        mean = np.mean(amp, axis=0, keepdims=True)
        std = np.std(amp, axis=0, keepdims=True)
        amp_norm = (amp - mean) / (std + 1e-8)

        n_frames = len(amp_norm)
        if n_frames < window_size:
            print(f'[경고] window보다 짧아서 제외: {filepath} ({n_frames} frames)')
            continue

        for start in range(0, n_frames - window_size + 1, stride_size):
            win = amp_norm[start:start + window_size]
            if len(win) == window_size:
                windows.append(win)

    if not windows:
        return np.empty((0, window_size, expected_feature_dim), dtype=np.float32)

    return np.asarray(windows, dtype=np.float32)


# ============================================================
# 3. 모델
# ============================================================
def build_1d_cnn(input_shape, num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_lstm(input_shape, num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ============================================================
# 4. 시나리오 분할
# ============================================================
def make_split(files_all, target):
    """
    중간점검용 고정 설계.

    A/B/C는 각각 20개를 모두 활용:
      Auth   : Train 16 / Test 4
      Unauth : 다른 두 사람 각각 Train 8 + 8 / Test 2 + 2

    D는 현재 4개만 활용:
      Empty  : Train 3 / Test 1

    예: A가 인가자
      Auth   = A01~A16 / A17~A20
      Unauth = B01~B08 + C01~C08 / B17~B18 + C17~C18
      Empty  = D01~D03 / D04
    """
    for person in ['A', 'B', 'C']:
        if len(files_all[person]) < 20:
            raise ValueError(
                f'{person} 데이터가 20개 필요합니다. 현재 {len(files_all[person])}개입니다.'
            )

    if len(files_all['D']) < 4:
        raise ValueError(
            f'D 데이터가 최소 4개 필요합니다. 현재 {len(files_all["D"])}개입니다.'
        )

    unauth_keys = [k for k in ['A', 'B', 'C'] if k != target]

    train_auth = files_all[target][:16]
    test_auth = files_all[target][16:20]

    # 비인가자는 두 사람을 동일 비율로 사용
    train_unauth = (
        files_all[unauth_keys[0]][:8]
        + files_all[unauth_keys[1]][:8]
    )
    test_unauth = (
        files_all[unauth_keys[0]][16:18]
        + files_all[unauth_keys[1]][16:18]
    )

    train_empty = files_all['D'][:3]
    test_empty = files_all['D'][3:4]

    return {
        'train_auth': train_auth,
        'test_auth': test_auth,
        'train_unauth': train_unauth,
        'test_unauth': test_unauth,
        'train_empty': train_empty,
        'test_empty': test_empty,
    }


# ============================================================
# 5. 평가/저장
# ============================================================
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(7, 5.5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title, fontsize=14, pad=12, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
    plt.yticks(tick_marks, CLASS_NAMES)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > threshold else 'black',
                fontweight='bold',
            )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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
def main():
    files_all = {
        'A': find_matching_files('A'),
        'B': find_matching_files('B'),
        'C': find_matching_files('C'),
        'D': find_matching_files('D'),
    }

    print('\n[파일 로드 상태]')
    for key in ['A', 'B', 'C', 'D']:
        print(f'  {key}: {len(files_all[key])}개')

    print('\n[중간점검 실험 설계]')
    print('  A/B/C : Train 16 / Test 4')
    print('  Unauth: 다른 두 사람에서 Train 8+8 / Test 2+2')
    print('  D     : Train 3 / Test 1  (임시)')
    print('  ※ D=20 수집 후에는 Empty도 Train 16 / Test 4로 변경 필요')

    # 필요한 전체 파일에서 공통 CSI feature dimension 결정
    all_used_groups = [
        files_all['A'][:20],
        files_all['B'][:20],
        files_all['C'][:20],
        files_all['D'][:4],
    ]
    feature_dim = determine_global_feature_dim(all_used_groups)

    summary_rows = []
    class_rows = []

    cm_total = {
        '1D-CNN': np.zeros((3, 3), dtype=int),
        'LSTM': np.zeros((3, 3), dtype=int),
    }

    for target in SCENARIOS:
        print(f'\n========== [{target}가 인가자일 때] ==========')
        split = make_split(files_all, target)

        print(
            f"Session | Train Auth={len(split['train_auth'])}, "
            f"Unauth={len(split['train_unauth'])}, Empty={len(split['train_empty'])} | "
            f"Test Auth={len(split['test_auth'])}, "
            f"Unauth={len(split['test_unauth'])}, Empty={len(split['test_empty'])}"
        )

        X_tr_auth = parse_csi_to_windows(split['train_auth'], feature_dim)
        X_te_auth = parse_csi_to_windows(split['test_auth'], feature_dim)
        X_tr_unauth = parse_csi_to_windows(split['train_unauth'], feature_dim)
        X_te_unauth = parse_csi_to_windows(split['test_unauth'], feature_dim)
        X_tr_empty = parse_csi_to_windows(split['train_empty'], feature_dim)
        X_te_empty = parse_csi_to_windows(split['test_empty'], feature_dim)

        arrays = [
            X_tr_auth, X_tr_unauth, X_tr_empty,
            X_te_auth, X_te_unauth, X_te_empty,
        ]
        labels = [
            'Train Auth', 'Train Unauth', 'Train Empty',
            'Test Auth', 'Test Unauth', 'Test Empty',
        ]

        for name, arr in zip(labels, arrays):
            if len(arr) == 0:
                raise RuntimeError(
                    f'{target} 시나리오의 {name} window가 0개입니다. '
                    '파일명 또는 CSI 파싱 형식을 확인하세요.'
                )

        X_train = np.concatenate(
            [X_tr_auth, X_tr_unauth, X_tr_empty], axis=0
        )
        y_train = np.concatenate([
            np.zeros(len(X_tr_auth), dtype=np.int64),
            np.ones(len(X_tr_unauth), dtype=np.int64),
            np.full(len(X_tr_empty), 2, dtype=np.int64),
        ])

        X_test = np.concatenate(
            [X_te_auth, X_te_unauth, X_te_empty], axis=0
        )
        y_test = np.concatenate([
            np.zeros(len(X_te_auth), dtype=np.int64),
            np.ones(len(X_te_unauth), dtype=np.int64),
            np.full(len(X_te_empty), 2, dtype=np.int64),
        ])

        # train만 섞고 test는 원래 class 묶음 상태 유지해도 평가에는 영향 없음
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        print(
            f'Window | Train Auth={len(X_tr_auth)}, '
            f'Unauth={len(X_tr_unauth)}, Empty={len(X_tr_empty)} | '
            f'Test Auth={len(X_te_auth)}, '
            f'Unauth={len(X_te_unauth)}, Empty={len(X_te_empty)}'
        )

        # D가 3세션뿐이라 Empty가 무시되지 않도록 class weight 자동 적용
        present_classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=present_classes,
            y=y_train,
        )
        class_weight = {
            int(cls): float(w) for cls, w in zip(present_classes, weights)
        }
        print(f'Class weight: {class_weight}')

        input_shape = X_train.shape[1:]

        # ----------------------------------------------------
        # 1D-CNN
        # ----------------------------------------------------
        print(f'[{target}] 1D-CNN 학습 시작')
        cnn = build_1d_cnn(input_shape)
        cnn.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_weight=class_weight,
            verbose=0,
        )
        pred_cnn = np.argmax(cnn.predict(X_test, verbose=0), axis=1)

        cm_cnn = confusion_matrix(y_test, pred_cnn, labels=[0, 1, 2])
        cm_total['1D-CNN'] += cm_cnn
        plot_confusion_matrix(
            cm_cnn,
            f'1D-CNN - {target} Auth',
            f'cm_1d_cnn_{target}_auth.png',
        )

        summary, details = evaluate_model(
            y_test, pred_cnn, target, '1D-CNN'
        )
        summary_rows.append(summary)
        class_rows.extend(details)
        print_scenario_result(summary)

        # GPU/메모리 정리 후 LSTM
        del cnn
        tf.keras.backend.clear_session()

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------
        print(f'[{target}] LSTM 학습 시작')
        lstm = build_lstm(input_shape)
        lstm.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_weight=class_weight,
            verbose=0,
        )
        pred_lstm = np.argmax(lstm.predict(X_test, verbose=0), axis=1)

        cm_lstm = confusion_matrix(y_test, pred_lstm, labels=[0, 1, 2])
        cm_total['LSTM'] += cm_lstm
        plot_confusion_matrix(
            cm_lstm,
            f'LSTM - {target} Auth',
            f'cm_lstm_{target}_auth.png',
        )

        summary, details = evaluate_model(
            y_test, pred_lstm, target, 'LSTM'
        )
        summary_rows.append(summary)
        class_rows.extend(details)
        print_scenario_result(summary)

        del lstm
        tf.keras.backend.clear_session()

    # ========================================================
    # 7. A/B/C 평균
    # ========================================================
    df_summary = pd.DataFrame(summary_rows)
    df_class = pd.DataFrame(class_rows)

    avg_rows = (
        df_summary
        .groupby('Model', as_index=False)[
            ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1']
        ]
        .mean()
    )
    avg_rows.insert(0, 'Scenario', 'AVERAGE(A/B/C)')

    df_summary_final = pd.concat(
        [df_summary, avg_rows],
        ignore_index=True,
    )

    class_avg = (
        df_class
        .groupby(['Model', 'Class'], as_index=False)[
            ['Precision', 'Recall', 'F1', 'Support']
        ]
        .mean()
    )
    class_avg.insert(0, 'Scenario', 'AVERAGE(A/B/C)')

    df_class_final = pd.concat(
        [df_class, class_avg],
        ignore_index=True,
    )

    # 보기 좋게 반올림
    metric_cols = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1']
    df_summary_final[metric_cols] = df_summary_final[metric_cols].round(4)

    class_metric_cols = ['Precision', 'Recall', 'F1', 'Support']
    df_class_final[class_metric_cols] = df_class_final[class_metric_cols].round(4)

    # CSV 저장
    summary_csv = 'csi_3class_scenario_summary.csv'
    class_csv = 'csi_3class_class_metrics.csv'
    df_summary_final.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_class_final.to_csv(class_csv, index=False, encoding='utf-8-sig')

    # 전체 시나리오 누적 CM 저장
    plot_confusion_matrix(
        cm_total['1D-CNN'],
        '1D-CNN - Aggregated A/B/C',
        'cm_1d_cnn_aggregated.png',
    )
    plot_confusion_matrix(
        cm_total['LSTM'],
        'LSTM - Aggregated A/B/C',
        'cm_lstm_aggregated.png',
    )

    print('\n========== [최종 요약] ==========')
    print(df_summary_final.to_string(index=False))

    print('\n[A/B/C 평균]')
    for _, row in avg_rows.iterrows():
        print(
            f"  {row['Model']:<7} | "
            f"Accuracy={row['Accuracy']:.4f} | "
            f"Macro F1={row['Macro_F1']:.4f}"
        )

    print('\n[저장 파일]')
    print(f'  - {summary_csv} : 시나리오별 성능 + 모델별 A/B/C 평균')
    print(f'  - {class_csv} : 클래스별 Precision/Recall/F1 + 평균')
    print('  - cm_1d_cnn_A_auth.png / B / C')
    print('  - cm_lstm_A_auth.png / B / C')
    print('  - cm_1d_cnn_aggregated.png')
    print('  - cm_lstm_aggregated.png')

    print('\n※ 현재 D는 4세션뿐인 중간점검용입니다.')
    print('※ 최종 실험에서는 D도 20세션 수집 후 16/4로 맞추세요.')


if __name__ == '__main__':
    main()
