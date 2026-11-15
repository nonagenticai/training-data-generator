#!/usr/bin/env bash
# Production entrypoint for training-data-generator when run as a host
# process under systemd-user on pgx. Invoked by
# gitops-production/scripts/host-configs/pgx/systemd/training-data-generator.service.
# Sibling pattern: pytorch_backend / pytorch_ingestion_backend / model-trainer.
#
# Reads runtime config from $ENV_FILE (defaults to
# $XDG_CONFIG_HOME/training-data-generator/env, falls back to
# /etc/training-data-generator/env). Source-of-truth for that file is the
# gitops ConfigMap (when wired up); until then it's a hand-managed file
# on pgx (mode 600).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/training-data-generator/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [ -f /etc/training-data-generator/env ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/training-data-generator/env
  set +a
fi

# uv on PATH (installer drops it at ~/.local/bin/uv).
export PATH="${HOME}/.local/bin:${PATH}"

exec uv run uvicorn main:app \
  --app-dir src \
  --host "${TRAINING_DATA_GEN_BIND_HOST:-0.0.0.0}" \
  --port "${TRAINING_DATA_GEN_PORT:-8152}"
