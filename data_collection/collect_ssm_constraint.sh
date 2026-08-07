#!/usr/bin/env bash
set -euo pipefail

# Matrix-rich trajectory for testing the SSM insertion/interface constraint
# away from the straight configuration. Tendon plateaus are established before
# insertion/rotation probes, and episode labels distinguish settled and dynamic
# portions in the downstream diagnostic.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd -- "${SIM_DIR}/.." && pwd)"
PYTHON_BIN="${CR_PYTHON:-${WORKSPACE_DIR}/cr-venv/bin/python}"
OUTPUT_DIR="${SOFA_OUTPUT_DIR:-${WORKSPACE_DIR}/sofa_data/ssm_constraint}"
SCENES_FILE="${SOFA_SCENES_FILE:-${SIM_DIR}/configs/freespace_full.yaml}"
SCENE_INDEX="${SOFA_SCENE_INDEX:-1}"

mkdir -p "${OUTPUT_DIR}"

COLLECT_GENERATOR=ssm_constraint \
COLLECT_MATRICES=1 \
COLLECT_COMPONENTS=1 \
COLLECT_DEBUG=1 \
COLLECT_A_INTERVAL="${COLLECT_A_INTERVAL:-5}" \
COLLECT_VERBOSE="${COLLECT_VERBOSE:-0}" \
"${PYTHON_BIN}" "${SIM_DIR}/scenes/collect_data.py" \
    --scenes "${SCENES_FILE}" \
    --scene-idx "${SCENE_INDEX}" \
    --output-dir "${OUTPUT_DIR}"
