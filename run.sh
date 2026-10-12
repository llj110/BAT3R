#!/usr/bin/env bash
# Compatibility launcher. Prefer bat3r_run.sh.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/bat3r_run.sh" "$@"
