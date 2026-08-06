#!/usr/bin/env bash
set -euo pipefail

# Component-resolved privileged SOFA verification trajectory.
# In every episode containing "zero_tendon", the commanded tendon FORCE is
# exactly zero. The load/hold/unload episodes explicitly contain a nonzero-force
# hold followed by a zero-force passive ring-down.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$SIM_DIR/.." && pwd)"
PYTHON_BIN="${CR_PYTHON:-$WORKSPACE_DIR/cr-venv/bin/python}"
OUTPUT_DIR="${SOFA_OUTPUT_DIR:-$WORKSPACE_DIR/sofa_data/proximal_identification}"
SCENES_FILE="${SOFA_SCENES_FILE:-$SIM_DIR/configs/freespace_full.yaml}"
SCENE_INDEX="${SOFA_SCENE_INDEX:-1}"

mkdir -p "$OUTPUT_DIR"

COLLECT_GENERATOR=proximal_id \
COLLECT_MATRICES=1 \
COLLECT_COMPONENTS=1 \
COLLECT_DEBUG=1 \
COLLECT_A_INTERVAL="${COLLECT_A_INTERVAL:-10}" \
COLLECT_VERBOSE="${COLLECT_VERBOSE:-0}" \
"$PYTHON_BIN" "$SIM_DIR/scenes/collect_data.py" \
    --scenes "$SCENES_FILE" \
    --scene-idx "$SCENE_INDEX" \
    --output-dir "$OUTPUT_DIR"
