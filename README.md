# zero_trust_csi

CSI 데이터로 Auth / Unauth / Empty를 분류하는 1D-CNN 및 LSTM 실험.

## 실행 환경: WSL2 / Ubuntu 24.04 / NVIDIA GPU

배포판은 `Ubuntu-24.04`, 일반 사용자는 `owner`로 설정한다.
프로젝트 코드는 Windows 폴더에 두고, 가상환경은 Ubuntu 파일 시스템의
`/home/owner/.venvs/zero_trust`에 설치한다.

2026-10-07 검증: WSL2, Ubuntu 24.04.5 LTS, Python 3.12.3,
TensorFlow 2.21.0, NVIDIA GeForce RTX 4060(8GB).
패키지 의존성, 한글 폰트, GPU 행렬 및 합성곱 연산, XLA 컴파일을 확인했다.
A/B/C/D 세션 파일은 각각 20개 검색됨을 확인했다. CSI 학습은 실행하지 않았다.

PowerShell 또는 Git Bash에서 Ubuntu 터미널을 연다:

```powershell
wsl -d Ubuntu-24.04 -u owner
```

Ubuntu 터미널에서 프로젝트 폴더로 이동하고 가상환경을 활성화한다:

```bash
cd /mnt/c/Users/OWNER/Desktop/newfloder/zero_trust
source scripts/activate_ubuntu.sh
python scripts/check_environment.py
```

검사 스크립트는 패키지, 한글 폰트, GPU 행렬 및 합성곱 연산, XLA 컴파일을
확인한다. 학습은 실행하지 않는다.
검사를 통과한 뒤 학습이 필요할 때 `python cha_gpt.py`로 실험한다.
데이터는 `data/A/A01.txt`부터 `data/D/D20.txt`까지 각 폴더에 둔다.
## 측정 및 데이터 분할

A/B/C는 사람 ID이며, 각 파일은 3분 측정 세션이다.
측정 시작부터 0~60초는 서서 정지, 60~120초는 앉아서 정지,
120~180초는 타자 치기로 구분한다. D는 3분 빈방 측정이며 행동 라벨은 `empty`다.
시간 기준점은 파일의 첫 번째 유효한 `pi_rx_time_ns` 값이다.
이 기준점과 실제 첫 행동 시작 시각이 일치한다고 가정한다.

인가자로 지정하는 사람을 A/B/C로 바꾸어 세 조건을 평가한다:

| 구분 | 인가자 | 나머지 두 사람 | 빈방 D | 합계 |
|---|---|---|---|---|
| 학습 | 01~16: 16개 | 각각 01~08: 8개씩 | 01~16: 16개 | 48개 |
| 테스트 | 17~20: 4개 | 각각 17~18: 2개씩 | 17~20: 4개 | 12개 |

비인가자일 때 각 사람의 09~16, 19~20은 그 조건에서 사용하지 않는다.
같은 사람의 17~20은 어느 조건에서도 학습에 사용하지 않는다.
각 파일의 윈도우 전체가 동일한 학습/테스트 분할에 속한다.

`csi_pipeline.py`가 시간 기준 전처리를 담당한다:

- I/Q 진폭을 계산하고 타임스탬프를 정렬하며 같은 시각의 중복 레코드를 제거한다.
- 표준 Espressif 로그의 `len`과 실제 CSI 길이를 대조한다.
  `first_word=1`이면 무효인 앞 4바이트를 0으로 처리한다.
  특징 인덱스가 이동하지 않도록 192차원은 유지하고 앞 2개 진폭 특징만 0으로 만든다.
- 각 행동 구간 안에서 선형 보간으로 20 FPS에 맞춘다.
- 2초 윈도우를 1초 간격으로 생성한다. 행동 경계를 넘는 윈도우는 만들지 않는다.
- 유효한 CSI 사이에 0.5초 초과 공백이 있으면 그 공백의 보간값을 포함하는 윈도우를 제외한다.
- 외삽하지 않으므로 각 구간의 시작/끝에 데이터가 부족하면 해당 윈도우를 만들지 않는다.
- 특징 차원 및 평균/표준편차는 해당 조건의 학습 데이터로 결정한다.
- 테스트 데이터에는 학습 평균/표준편차를 그대로 적용한다.
- 파일 내용의 SHA-256 중복을 검사하고 복사된 측정 파일이 있으면 중단한다.

학습 없이 실제 전처리만 확인하려면:

```bash
python cha_gpt.py --prepare-only
```

학습 후 전체/클래스별 성능 CSV와 혼동행렬 외에 다음 파일을 저장한다:

- `csi_activity_metrics.csv`: standing/sitting/typing/empty별 성능.
  사람의 세 행동은 Auth/Unauth를, 빈방은 Empty를 대상으로 Macro 지표를 계산한다.
  행동별 Accuracy는 Empty를 포함한 오분류도 오류로 계산한다.
- `csi_test_window_predictions.csv`: 사람, 원본 파일, 수신기, 행동, 시간 범위, 정답/예측 라벨.
- 예측 CSV에는 `Probability_Auth`, `Probability_Unauth`, `Probability_Empty`도 저장한다.
- `csi_training_history_1d_cnn_A.csv` 등: epoch별 학습 loss/accuracy.
- `csi_model_1d_cnn_A.keras` 등: 학습한 모델. 이후 오분류 분석 시 재학습 없이 불러올 수 있다.
- `csi_session_split.csv`: 인가자 조건별 실제 학습/테스트 파일 목록.
- `csi_preprocessing_A.npz` / B / C: 학습 정규화 통계 및 시간 전처리 설정.

혼동행렬의 행은 실제 클래스, 열은 예측 클래스이며 순서는
`Auth`(인가자), `Unauth`(비인가자), `Empty`(빈방)다.
혼동행렬의 건수는 테스트 파일에서 생성한 윈도우 수이며 파일 수가 아니다.
`cm_1d_cnn_A_auth.png` 등 기존 건수 이미지에 더해
`*_normalized.png`(행별 비율), `*_counts.csv`, `*_normalized.csv`를 저장한다.
비율 CSV는 0~1 값이며 이미지에는 백분율을 표시한다.
각 조건/모델 6개와 모델별 합산 2개, 총 8개 혼동행렬에 대해
PNG 16개와 CSV 16개를 저장한다. 실제 클래스가 없는 행은 비율을 0으로 표시한다.
합산 행렬은 세 조건의 테스트 윈도우 건수를 합한 값이다.

전처리 검증 모드에서는 학습이나 결과 파일 저장을 수행하지 않는다.
Optuna 탐색 코드는 별도 `tune_optuna.py`에 구현했다. 실제 탐색은 실행하지 않았다.

2026-10-07 최초 검사에서 발견한 A/D 16쌍의 동일 파일은 데이터 교체 후 해소됐다.
현재 80개 파일의 바이트 내용 및 타임스탬프를 제외한 CSI 값 전체에서
동일 기록이 없고, 기본 실험/Optuna 전처리도 통과했다.
파일별 검사는 `python scripts/audit_data.py`로 수행하며 `data_audit/`에 저장한다.
이 검사는 실제 사람 ID나 빈방 라벨의 정확성을 확정하지 않는다.

2026-10-07 코드 검증: 세션 분할, 시간 경계, 공백 처리, 정규화,
행동별 평가 및 학습 없는 검증 모드 테스트 13개를 통과했다.
실제 80개 파일에서도 `(40, 192)` 형태의 윈도우 생성과 유한한 진폭 값을
확인했다. 이 형식 검사는 중복 파일의 사람/빈방 라벨을 검증한 결과가 아니다.

```bash
python -m unittest discover -s tests -v
```

## Optuna 최적화: 고정 검증 세트

폴드 교차검증을 사용하지 않는다. 각 인가자 조건의 원래 학습 48개만 사용해
학습 36개와 검증 12개로 나눈다. 원래 최종 테스트 12개는 최적화에 사용하지 않는다.

| 용도 | 인가자 | 나머지 두 사람 | 빈방 D | 합계 |
|---|---|---|---|---|
| 최적화 학습 | 01~12: 12개 | 각각 01~06: 6개씩 | 01~12: 12개 | 36개 |
| 최적화 검증 | 13~16: 4개 | 각각 07~08: 2개씩 | 13~16: 4개 | 12개 |
| 최종 테스트 예약 | 17~20: 4개 | 각각 17~18: 2개씩 | 17~20: 4개 | 12개 |

검증 Macro F1을 최대화하는 TPE 탐색을 수행한다. 학습 정규화 통계는
최적화 학습 36개로만 계산한다. 탐색 범위는 `tune_optuna.py`의 `SEARCH_SPACE`에 있다.
최대 60 epoch이며 검증 Macro F1이 5회 개선되지 않으면 학습을 중단하고
최선의 가중치를 복원한다. MedianPruner로 가능성이 낮은 Trial을 중단한다.
GPU 한 개에서 Trial을 순서대로 실행한다.

현재 Ubuntu 가상환경에는 Optuna 5.0.0을 설치했다. 다른 환경에서 추가하려면:

```bash
source scripts/activate_ubuntu.sh
python -m pip install -r requirements-tuning.txt
```

최적화 코드 추가 후 테스트 22개와 패키지 의존성 검사를 통과했다.
테스트는 CPU 모델 구성과 임시 SQLite 기록 검사를 포함하며 실제 학습/탐색을 수행하지 않는다.

A 인가자 조건의 CNN만 20회 탐색하려면:

```bash
python tune_optuna.py --targets A --models cnn --trials 20
```

세 인가자 조건과 두 모델을 모두 탐색하려면:

```bash
python tune_optuna.py --trials 20
```

기본값은 6개 Study에 각각 20회, 총 120회 Trial이다.
같은 명령을 다시 실행하면 Study마다 20회를 추가한다.
진행 기록은 Ubuntu의 `~/.local/share/zero_trust/optuna/*.sqlite3`에 저장한다.
데이터/탐색 설정이 바뀌면 기존 Study에 합치지 않고 오류를 표시한다.
새 조건은 `--storage-dir`과 `--output-dir`에 새 폴더를 지정하여 실행한다.
전처리 버전도 실험 조건 해시에 포함한다. 무효 바이트 처리를 추가한 버전 2는
기존 버전 1의 Optuna Trial과 합치거나 이전 최적화 JSON으로 최종 평가하지 않는다.

완료된 Study의 결과는 프로젝트의 `optuna_results/`에 저장한다:

- `A_cnn_best.json` 등: 최적 설정, 검증 Macro F1, 비인가 오인 비율, 실험 조건 해시,
  최적 Trial의 검증 Macro F1이 가장 높았던 `best_epoch`, 최적화 `max_epochs`.
- `A_cnn_trials.csv` 등: Trial별 설정, 성능, 완료/실패/가지치기 상태.

최적화 결과는 검증 성능이다. 최종 성능은 선택한 설정으로 학습 48개 전체를
재학습하고, 예약한 최종 테스트 12개를 평가하여 구한다.

### 최적화 후 최종 테스트 혼동행렬

`evaluate_optuna.py`는 최적 설정 JSON을 읽고 현재 학습 데이터 및 탐색 조건이
일치하는지 검사한 뒤, 학습 48개로 다시 정규화 통계를 계산하고 재학습한다.
재학습 epoch 수는 최적 Trial의 검증 성능으로 선택한 `best_epoch`다.
최종 테스트 12개는 epoch 선택이나 조기 종료에 사용하지 않고 재학습 완료 후 평가한다.
최종 테스트 결과를 보고 설정을 다시 선택하면 해당 테스트는 독립적인 최종 평가가 아니게 된다.

A 조건의 CNN 최적화를 끝낸 뒤:

```bash
python -u evaluate_optuna.py --targets A --models cnn
```

A/B/C의 CNN과 LSTM 최적화 6개를 모두 끝낸 뒤:

```bash
python -u evaluate_optuna.py
```

`optuna_evaluation/`에 다음 결과를 저장한다:

- `cm_A_cnn_final_test.png`: 건수 혼동행렬.
- `cm_A_cnn_final_test_normalized.png`: 실제 클래스별 백분율 혼동행렬.
- `cm_A_cnn_final_test_counts.csv` / `*_normalized.csv`: 해당 행렬의 수치.
- 여러 인가자 조건을 함께 평가하면 모델별 합산 혼동행렬도 저장한다.
- `A_cnn_test_predictions.csv`, `A_cnn_preprocessing.npz`: 테스트 예측 및 재학습 정규화 설정.
- `A_cnn_train_predictions.csv` 등: 학습 윈도우의 예측과 클래스별 확률.
- `A_cnn_train_session_metrics.csv` / `A_cnn_test_session_metrics.csv` 등:
  파일별 정확도, 각 클래스 판정 건수 및 평균 예측 확률.
- `A_cnn_model.keras`, `A_cnn_training_history.csv` 등: 재학습 모델과 epoch별 기록.
- 테스트 예측 CSV에도 `Probability_Auth`, `Probability_Unauth`, `Probability_Empty`를 저장한다.
- `final_train_summary.csv`, `final_train_class_metrics.csv`: 학습 데이터의 성능.
- `final_test_summary.csv`, `final_test_class_metrics.csv`, `final_test_activity_metrics.csv`: 최종 성능.
- `evaluation_manifest.json`: 사용한 최적 Trial, 설정, epoch 및 학습/테스트 파일 목록과 내용 해시.

최적화 JSON이 없는 조건은 오류를 표시하므로, 완료된 조건만 `--targets`와
`--models`로 선택한다. 재학습 epoch를 직접 지정하려면 `--epochs 25`처럼 사용한다.
결과 파일을 만든 후 학습 없는 전처리 검증도 가능하다:

```bash
python evaluate_optuna.py --targets A --models cnn --prepare-only
```

최종 평가 코드는 작성 및 검증만 했으며 실제 재학습/최적화는 실행하지 않았다.
혼동행렬의 행/열 방향, 비율 계산, 최종 테스트 분리 및 epoch 선택을 포함해
전체 테스트 28개를 통과했다. 예시 값으로 PNG 표시도 확인했다.

### 빈방 50% 결과의 코드 점검

2026-10-07 저장된 예측 12,626행을 실제 전처리와 대조했다.
사람/빈방 라벨 불일치가 0개이고, 테스트 윈도우 순서와 모든 조건별/합산 혼동행렬
건수가 일치했다. 당시 정규화 통계도 저장된 NPZ와 같았다.
결과는 `data_audit/code_review.json`에 있으며 무효 바이트 수정 전 코드의 검사다.

점검 중 표준 로그의 `first_word=1` 표시를 무시하여 무효인 앞 4바이트를
학습에 포함하던 문제를 수정했다. 근거는
[Espressif CSI 문서](https://docs.espressif.com/projects/esp-idf/en/v4.4.8/esp32/api-guides/wifi.html#wifi-channel-state-information)다.
이 문제가 빈방 50%의 단독 원인이었는지는 아직 확인하지 않았다.
기존 파일별 정규화에서 학습 데이터 기준 정규화로 변경한 영향도 별도 비교가 필요하다.
당시 점검에서는 재학습을 실행하지 않았으므로 그때의 혼동행렬은 수정 전 결과다.
수정 후 테스트 33개가 통과했고, 실제 80개 파일을 대상으로
`python cha_gpt.py --prepare-only` 전처리 검증도 완료했다.

### D01~D20 전체 검사

`python scripts/audit_empty_all.py`는 모델 학습 없이 현재 원본 80개의 중복을 검사하고,
D20개 전체의 패킷/메타데이터/측정 시각/윈도우/신호 분포를 비교한다.
결과는 `data_audit/D_all_files_audit.csv`, `D_all_files_audit.json`,
`D_all_profiles.png`에 저장한다. 입력 데이터와 기존 학습 결과는 수정하지 않는다.
현재 D01~D04 로그 시각은 10월 3일, D05~D18은 10월 6일,
D19~D20은 10월 7일이다. D19~D20은 이전 `recordings.csv` 검사 후 변경된 파일이다.
학습용 D01과 D02~D04도 D05~D16과 다른 기본 신호 패턴을 보인다.
현재 D17~D18은 학습용 D13~D16과 유사하고 D19~D20은 별도 패턴이다.
이 차이는 측정 대상 라벨의 오류나 모델 학습 실패를 확정하는 근거는 아니다.
기존 결과에는 학습용 D의 예측 및 입력 파일 해시가 없어 학습용 빈방 정확도나
현재 파일과 기존 예측의 내용 일치 여부까지 검증할 수는 없다.

### 2026-10-07 Optuna 최종 평가 점검

사용자가 Optuna 120회와 최종 평가를 실행한 후,
`python scripts/diagnose_optuna_results.py`로 저장 결과를 다시 점검했다.
6개 조건 모두 현재 학습 데이터의 실험 조건 해시, 정규화 통계,
테스트 라벨/윈도우 시각 및 혼동행렬 건수가 일치했다. 이 점검은 재학습하지 않는다.
검증 Macro F1은 0.9017~0.9822였지만 최종 빈방 재현율은
5개 조건에서 0.5, C CNN에서 0.4972였다. D19~D20은 모든 조건에서 사람으로 오분류됐다.
D01~D18은 오전 검사와 내용 해시가 같고, 교체된 D19~D20 로그 시각은
18:06/18:10이다. 두 파일은 최적화 학습/검증에 포함되지 않는다.
현재 D17~D20 RSSI 중앙값은 모두 -54 dBm이지만, D19~D20의 서브캐리어별
진폭 패턴은 학습용 D와 다르다. 측정 조건 변화에 대한 일반화 실패가 유력한 가설이며,
학습 부족이나 정규화의 기여를 확정하려면 학습용 예측과 별도의 통제 비교가 필요하다.
점검 자료는 `data_audit/optuna_empty_diagnosis_20261007/`에 저장한다.

후속 코드 점검에서는 최선 가중치 복원, epoch 선택, 날짜 불변성 및
학습/테스트 진단 저장을 포함한 테스트 38개가 통과했다.
최종 평가가 학습용 예측과 모델을 보관하지 않던 진단 공백을 보완했으며,
기존 최적화 결과 6개와의 호환성을 확인했다. 모델 학습 방식과 고정 검증 분할은 그대로다.
이 보완을 위해 Optuna를 다시 실행할 필요는 없다. 진단 자료를 생성하려면
기존 최적 설정으로 최종 재학습만 실행하고, 이전 결과와 비교할 별도 폴더를 지정한다:

```bash
python -u evaluate_optuna.py --best-dir optuna_results_20261007_new --output-dir optuna_evaluation_diagnostics_20261007
```

학습용 D의 정확도도 낮으면 학습/표현 문제를, 학습용 D는 잘 맞고 새 측정일 D만
실패하면 일반화 문제를 우선 조사한다. 이 구분만으로 원인이 완전히 확정되지는 않는다.

기존 결과를 다시 대조하려면 `python scripts/check_experiment_results.py`를 사용한다.
데이터나 전처리가 달라지면 이전 결과와 정규화 통계가 일치하지 않을 수 있다.

학습 없이 최적화용 전처리만 확인하려면:

```bash
python tune_optuna.py --prepare-only
```

새 Ubuntu 터미널에서도 프로젝트 폴더로 이동하고 위 `source` 명령으로
가상환경을 다시 활성화한다. 이 스크립트는 pip로 설치된 NVIDIA 라이브러리의
`LD_LIBRARY_PATH`와 XLA용 CUDA 데이터 경로도 현재 터미널에 설정한다.

`requirements.txt`는 기본 패키지 목록이고,
`requirements-gpu.txt`는 Linux / WSL2용 CUDA 의존성을 추가한다.
Windows와 WSL 사이에서 같은 가상환경을 공유하지 않는다.

## 대상별 15개 비교 실험

`--sessions-per-person 15`는 A/B/C/D의 번호순 01~15만 선택한다.
원본 16~20은 삭제하지 않으며 기존 20개 실험은 기본값으로 유지한다.
각 인가자 조건에서 Auth/Unauth/Empty는 최종 학습 12/12/12,
최종 테스트 3/3/3으로 구성된다. 비인가자는 사람별 학습 6개다.
테스트 1.5개씩은 불가능하므로 A 인가 시 B 1/C 2,
B 인가 시 A 2/C 1, C 인가 시 A 1/B 2개를 배정한다.
세 조건을 합하면 각 사람이 비인가자 테스트에 3개씩 참여한다.

Optuna 고정 검증 분할은 최종 학습 36개에서 학습 27개/검증 9개다.
각 클래스는 학습 9개/검증 3개이며, 최종 테스트는 최적화에 사용하지 않는다.
Fold 없이 CNN/LSTM × A/B/C의 6개 Study를 새로 최적화한다.
파일 선택 및 모든 분할은 `selection_manifest.json`에 보관한다.

```bash
python -u tune_optuna.py --sessions-per-person 15 --trials 20 --max-epochs 60 --storage-dir /home/owner/.local/share/zero_trust/optuna_15_20261007 --output-dir optuna_results_15_20261007
python -u evaluate_optuna.py --sessions-per-person 15 --best-dir optuna_results_15_20261007 --output-dir optuna_evaluation_15_20261007
```

번호순으로 제외하면 기존에 오분류된 D19~D20도 테스트에서 빠지고,
빈방 테스트가 D13~D15로 바뀐다. 성능 상승만으로 데이터 감소의 효과나
다른 날짜의 빈방에 대한 일반화 개선을 확정할 수 없다.
필요한 추가 진단은 저장 모델로 제외된 D16~D20을 예측한다.
이 예측으로 하이퍼파라미터를 선택하거나 모델을 다시 학습하지 않는다.

```bash
python -u scripts/evaluate_excluded_empty.py --evaluation-dir optuna_evaluation_15_20261007 --output-dir data_audit/excluded_empty_15_20261007
```

2026-10-07~08 실제 실행 결과: 120개 Trial 중 정상 완료 92개,
가지치기 28개, 실패 0개다. 모델별 최종 테스트는 다음과 같다.
빈방 재현율은 D13~D15의 윈도우 중 Empty로 맞힌 비율이다.

| 인가자 | 모델 | Macro F1 | 빈방 F1 | 빈방 재현율 |
| --- | --- | ---: | ---: | ---: |
| A | 1D-CNN | 0.4507 | 0.0000 | 0.00% |
| A | LSTM | 0.5000 | 0.0037 | 0.19% |
| B | 1D-CNN | 0.8482 | 0.9962 | 99.24% |
| B | LSTM | 0.3922 | 0.0401 | 2.05% |
| C | 1D-CNN | 0.8700 | 0.9962 | 99.24% |
| C | LSTM | 0.4339 | 0.0178 | 0.95% |

6개 조건 모두 학습용 빈방 재현율은 100%였다.
저장 모델로 추가 예측한 D19~D20의 빈방 재현율은 6개 조건 모두 0%였다.
A의 두 모델과 B/C의 LSTM은 D13~D18에서도 대부분 실패했다.
따라서 파일 수 감소가 빈방 판별 문제를 해결하지 않았고,
10월 7일 파일만으로 전체 실패를 설명할 수도 없다.
측정 회차별 신호 차이와 학습 데이터의 대표성을 함께 조사해야 한다.

최종 결과는 `optuna_evaluation_15_20261007/`에 저장했다.
6개 혼동행렬을 모은 `confusion_matrices_overview.png`도 포함한다.
추가 D 진단은 `data_audit/excluded_empty_15_20261007/`,
20개/15개 비교표와 실제 전처리 대조는
`data_audit/reduced_experiment_20261007/`에 보관한다.
현재 원본 80개의 해시가 시작 시점과 같고, 6개 조건의 실험 조건,
정규화 통계, 테스트 라벨/시간, 혼동행렬 및 지표가 실제 전처리·예측과 일치했다.
분할 및 기존 호환성을 포함한 단위 테스트 42개도 통과했다.
20개/15개 실험은 테스트 파일과 최적 설정이 모두 달라졌으므로,
성능 차이를 파일 개수 감소의 단독 효과로 해석하지 않는다.

## 학습 실행 재현성

### 2026-10-08 학습 재현성 점검 및 수정

위 15개 표는 결정론적 연산을 활성화하기 전의 한 번의 실행 결과다.
그 결과만으로 데이터 차이를 분류 실패의 확정 원인으로 해석하지 않는다.
같은 A CNN 설정, 같은 데이터, 같은 seed 42, 같은 초기 가중치로
단독 GPU 재학습을 두 번 했는데 빈방 재현율이 0%와 100%로 달라졌다.
원본 파싱 오류, 파일→숫자 라벨 매핑 오류 및 GPU 예측 오류는 재현되지 않았다.
CSV 모듈로 독립 디코딩한 D01~D15와 시간 윈도우가 원래 전처리와 일치했고,
저장 모델의 CPU 예측은 기존 GPU 예측 1,536개와 모두 같은 판정이었다.
CPU에서 새로 학습한 모델의 빈방 재현율은 48.11%였다.

`tf.keras.utils.set_random_seed`만으로 GPU 학습 결과가 고정되지 않으므로,
`tf.config.experimental.enable_op_determinism()`을 기본 학습 실행에 추가했다.
[TensorFlow 공식 설명](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)에
따라 같은 하드웨어·소프트웨어에서 seed와 결정론적 연산을 함께 설정한다.
이 설정으로 재학습한 A CNN 두 번은 모든 테스트 예측 확률이 정확히 같았고,
빈방 재현율도 두 번 모두 0.76%였다. 재현성 수정은 분류 성능 향상을 보장하지 않는다.

`tune_optuna.py`, `evaluate_optuna.py`는 이제 결정론적 연산이 기본값이다.
이전 동작의 비교 실험에만 `--no-deterministic`을 사용한다.
새 Optuna 실행은 이전 Study와 섞이지 않도록 실행 설정을 조건 해시에 포함한다.
새 저장 폴더를 사용해야 한다. 이전 최적 JSON의 숫자 설정은 재학습에 재사용할 수 있지만,
최적화와 재학습의 실행 설정이 다르면 이를 출력하고 평가 manifest에 각각 기록한다.
기존 최적화 JSON에 설정 항목이 없으면 이전 비결정론적 실행으로 취급한다.
기존 결과 파일은 수정하지 않았다. 수정 후 테스트 44개가 통과했다.

진단 자료는 `data_audit/execution_integrity_15_20261008/`에 보관한다.
`execution_integrity.json`, `gpu_standard_repeats.json`,
`gpu_deterministic_repeats.json`에 각 비교의 실제 결과가 기록되어 있다.
재현하려면 다음을 각각 별도 Python 프로세스로 실행한다.

```bash
python -u scripts/check_execution_integrity.py --cpu-refit
python -u scripts/check_training_repeatability.py --runs 2
python -u scripts/check_training_repeatability.py --runs 2 --deterministic
```

## 원본 파일 단위 무작위 분할 비교

20개 원본 전체에서 분할 seed `0, 1, 2, 3, 4`를 비교한다.
번호순 분할도 같은 결정론적 연산 설정으로 다시 학습해 비교 기준으로 삼는다.
인가자 A/B/C별 CNN/LSTM에 대해 총 36회 학습한다. Fold는 적용하지 않는다.
각 사람의 파일 목록을 먼저 무작위로 섞고, 학습 풀 16개와 테스트 풀 4개로 나눈다.
인가자와 빈방은 각각 학습 16개/테스트 4개를 사용한다.
비인가자 두 사람은 학습 풀에서 각각 8개, 테스트 풀에서 각각 2개를 사용한다.
각 실험의 전체 학습은 48개, 테스트는 12개로 기존 클래스 비율을 유지한다.
같은 원본 파일의 윈도우는 학습과 테스트에 나뉘지 않는다.
같은 분할 seed에서 A/B/C 실험은 사람별 학습/테스트 풀을 공유한다.

이전 20개 실험의 `optuna_results_20261007_new/` 설정과 best_epoch를 그대로 사용한다.
Optuna를 다시 실행하지 않고, 테스트를 보고 epoch를 선택하지 않는다.
학습 seed는 42, 결정론적 연산은 활성화하고 분할 seed만 바꾼다.
원본 파일 단위 학습 정규화는 각 새 학습 분할에서 다시 계산한다.

```bash
source scripts/activate_ubuntu.sh
python -u scripts/compare_random_splits.py --seeds 0 1 2 3 4 --output-dir random_split_evaluation_20_20261008
```

기존 결과를 덮어쓰지 않으므로 재실행하려면 새 `--output-dir`을 지정한다.
하나의 조건만 실행하려면 `evaluate_optuna.py`에 `--split-seed 0`을 추가한다.
최적 JSON의 입력 해시는 원래 번호순 최적화 분할과 먼저 검증한다.
새 분할마다 최적화 당시 사용했던 파일과 새 테스트의 교집합을 manifest에 기록한다.
새 테스트가 과거 최적화 데이터와 겹칠 수 있어 이 실험은 분할 민감도 탐색이다.
하이퍼파라미터까지 독립적으로 평가하는 실험은 분할별 학습 내부에서 다시 최적화해야 한다.
여러 seed에서 같은 원본이 반복 평가되므로 통합 혼동행렬은 독립 표본 개수로 해석하지 않는다.

결과 폴더에는 `ordered/`, `seed_0/`~`seed_4/`의 모델·확률·혼동행렬,
`all_results.csv`, `random_seed_statistics.csv`, `split_comparison.png`,
`cm_cnn_random_pooled.png`, `cm_lstm_random_pooled.png`가 생성된다.
`experiment_manifest.json`은 설정/파일 분할/해시, `progress.json`은 실행 상태를 저장한다.

2026-10-08 실제 실행을 약 19분에 완료했다. 번호순 6개와 무작위 30개 모델을
모두 정상 학습했다. 분할 검증을 포함한 테스트 45개가 통과했고,
최종 대조에서 원본 80개 해시가 동일하고 학습/테스트 파일 중복이 없었다.
36개 모델의 설정, epoch 수, 파일별 라벨/확률, 혼동행렬 및 지표도 일치했다.
`verification.json`에 검증 결과를 보관한다.

| 인가자 | 모델 | 번호순 Macro F1 | 무작위 Macro F1 평균 ± 표준편차 | 무작위 빈방 재현율 평균 |
| --- | --- | ---: | ---: | ---: |
| A | 1D-CNN | 0.6468 | 0.9026 ± 0.0669 | 94.94% |
| A | LSTM | 0.6617 | 0.8942 ± 0.0700 | 95.00% |
| B | 1D-CNN | 0.6804 | 0.8964 ± 0.0684 | 94.97% |
| B | LSTM | 0.6483 | 0.8967 ± 0.0807 | 94.97% |
| C | 1D-CNN | 0.5762 | 0.9212 ± 0.0656 | 95.00% |
| C | LSTM | 0.5629 | 0.9118 ± 0.0663 | 95.00% |

번호순 빈방 재현율은 C CNN 49.72%, 나머지 50%였다.
무작위 seed 0은 6개 모두 75%, seed 1은 99.72~100%, seed 2/3/4는 모두 100%였다.
seed 0의 D01은 6개 모델이 모두 실패했다. D20은 번호순에서 모두 실패했지만
seed 4에서 모두 성공했다. seed 4의 학습에는 D19가 포함된다.
따라서 데이터 구성에 따른 성능 차이는 확인되지만, 새 날짜까지의 일반화나
특정 파일의 실제 점유 라벨을 확정하는 근거로 삼지 않는다.
`empty_test_sessions.csv`에 원본 빈방 파일별 결과를 모았다.

## 환경 재설치

Ubuntu 24.04가 없는 컴퓨터에서는 관리자 PowerShell에서
`wsl --install -d Ubuntu-24.04 --no-launch`로 설치한다.
재부팅 안내가 있으면 재부팅한다. WSL 배포판의 VERSION은 2여야 한다.

이 프로젝트 위치에서 PowerShell로 환경 설치 스크립트를 실행한다:

```powershell
wsl -d Ubuntu-24.04 -u root --exec bash /mnt/c/Users/OWNER/Desktop/newfloder/zero_trust/scripts/setup_ubuntu.sh
```

스크립트는 Python 도구와 나눔 폰트를 설치하고, `owner` 계정과 가상환경을
만든 뒤 패키지 의존성을 확인한다. 학습 코드는 실행하지 않는다.
실제 설치 버전은 `requirements-wsl.lock.txt`에 기록한다.

`owner` 계정에는 자동으로 비밀번호나 sudo 권한을 부여하지 않는다.
관리 작업은 PowerShell에서 `wsl -d Ubuntu-24.04 -u root`로 수행할 수 있다.
Ubuntu 비밀번호가 필요한 경우 다음 명령으로 직접 설정한다:

```powershell
wsl -d Ubuntu-24.04 -u root --exec passwd owner
```
