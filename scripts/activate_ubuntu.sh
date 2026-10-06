# Source this file from Ubuntu: source scripts/activate_ubuntu.sh

_zero_trust_venv=/home/owner/.venvs/zero_trust
if [[ ! -x "$_zero_trust_venv/bin/python" ]]; then
    printf 'Python environment is missing: %s\n' "$_zero_trust_venv" >&2
    return 1
fi
source "$_zero_trust_venv/bin/activate"

_zero_trust_libs=$(python -c "import sysconfig; from pathlib import Path; p=Path(sysconfig.get_paths()['purelib'])/'nvidia'; print(':'.join(str(d) for d in sorted(p.glob('*/lib'))))")
export LD_LIBRARY_PATH="$_zero_trust_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

_zero_trust_cuda=$(python -c "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()['purelib'])/'nvidia/cuda_nvcc')")
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_cuda_data_dir="* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$_zero_trust_cuda"
fi
unset _zero_trust_venv _zero_trust_libs _zero_trust_cuda
