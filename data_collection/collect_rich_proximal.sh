#!/usr/bin/env bash
set -euo pipefail

# Rich, multi-episode trajectories for learning an observable mechanical latent.
# Different seeds change multisine phases but retain episode names and timing.
# Set SOFA_SEEDS="0" for one trajectory or COLLECT_A_INTERVAL=1 for dense A.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$SIM_DIR/.." && pwd)"
PYTHON_BIN="${CR_PYTHON:-$WORKSPACE_DIR/cr-venv/bin/python}"
OUTPUT_ROOT="${SOFA_OUTPUT_DIR:-$WORKSPACE_DIR/sofa_data/proximal_rich}"
SCENES_FILE="${SOFA_SCENES_FILE:-$SIM_DIR/configs/freespace_full.yaml}"
SCENE_INDEX="${SOFA_SCENE_INDEX:-1}"
SEEDS="${SOFA_SEEDS:-0 1 2}"

for seed in $SEEDS; do
    output_dir="$OUTPUT_ROOT/seed_$seed"
    mkdir -p "$output_dir"
    echo "[collect_rich_proximal] seed=$seed output=$output_dir"
    COLLECT_GENERATOR=rich_proximal \
    COLLECT_SEED="$seed" \
    COLLECT_MATRICES="${COLLECT_MATRICES:-1}" \
    COLLECT_COMPONENTS="${COLLECT_COMPONENTS:-1}" \
    COLLECT_DEBUG="${COLLECT_DEBUG:-1}" \
    COLLECT_A_INTERVAL="${COLLECT_A_INTERVAL:-10}" \
    COLLECT_VERBOSE="${COLLECT_VERBOSE:-0}" \
    "$PYTHON_BIN" "$SIM_DIR/scenes/collect_data.py" \
        --scenes "$SCENES_FILE" \
        --scene-idx "$SCENE_INDEX" \
        --output-dir "$output_dir"
done
