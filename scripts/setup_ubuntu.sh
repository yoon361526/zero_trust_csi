#!/usr/bin/env bash
set -euo pipefail

# Run as root in Ubuntu-24.04. Installs and checks the environment only.
if [[ $(id -u) != 0 ]]; then
    printf 'Run this setup using wsl -d Ubuntu-24.04 -u root.\n' >&2
    exit 1
fi

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
linux_user=owner
venv_dir=/home/owner/.venvs/zero_trust

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv fonts-nanum

if ! id "$linux_user" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$linux_user"
fi

# Keep existing WSL settings and select the ordinary account for new terminals.
python3 - <<'PY'
import configparser
from pathlib import Path

path = Path('/etc/wsl.conf')
config = configparser.ConfigParser()
config.read(path)
if not config.has_section('user'):
    config.add_section('user')
config.set('user', 'default', 'owner')
with path.open('w') as stream:
    config.write(stream)
PY

runuser -u "$linux_user" -- python3 -m venv "$venv_dir"
runuser -u "$linux_user" -- "$venv_dir/bin/python" -m pip install --upgrade pip
runuser -u "$linux_user" -- "$venv_dir/bin/python" -m pip install -r "$project_dir/requirements-gpu.txt"

# TensorFlow's documented fallback for pip CUDA library discovery in WSL.
runuser -u "$linux_user" -- "$venv_dir/bin/python" - <<'PY'
import os
import sys
import sysconfig
from pathlib import Path

packages = Path(sysconfig.get_paths()['purelib'])
tensorflow_dir = packages / 'tensorflow'
for library in (packages / 'nvidia').glob('*/lib/*.so*'):
    link = tensorflow_dir / library.name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(os.path.relpath(library, tensorflow_dir))

ptxas = packages / 'nvidia/cuda_nvcc/bin/ptxas'
link = Path(sys.prefix) / 'bin/ptxas'
if ptxas.exists() and not link.exists() and not link.is_symlink():
    link.symlink_to(os.path.relpath(ptxas, link.parent))
PY

runuser -u "$linux_user" -- "$venv_dir/bin/python" -m pip check
runuser -u "$linux_user" -- "$venv_dir/bin/python" -m pip freeze > "$project_dir/requirements-wsl.lock.txt"

printf '\nPython environment installed: %s\n' "$venv_dir"
