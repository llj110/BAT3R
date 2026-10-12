#!/usr/bin/env bash
# Portable BAT3R demo launcher; PYTHON may select an existing environment.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" bat3r_demo.py "$@"
