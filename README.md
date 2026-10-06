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
현재 Empty 분할은 D01~D03 학습, D04 테스트로 설정되어 있다.

새 Ubuntu 터미널에서도 프로젝트 폴더로 이동하고 위 `source` 명령으로
가상환경을 다시 활성화한다. 이 스크립트는 pip로 설치된 NVIDIA 라이브러리의
`LD_LIBRARY_PATH`와 XLA용 CUDA 데이터 경로도 현재 터미널에 설정한다.

`requirements.txt`는 기본 패키지 목록이고,
`requirements-gpu.txt`는 Linux / WSL2용 CUDA 의존성을 추가한다.
Windows와 WSL 사이에서 같은 가상환경을 공유하지 않는다.

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
