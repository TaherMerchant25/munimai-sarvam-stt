#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}:$(dirname "$0")/.."

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

