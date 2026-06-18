#!/bin/bash
# for i in 11 12 13 14 15 16 17; do
#     COLLECT_GENERATOR=sinusoidal COLLECT_DURATION=3600 \
#     python simulation/scenes/collect_data.py \
#         --scenes simulation/configs/generated_scenes.yaml \
#         --scene-idx $i \
#         --output-dir /media/chen-lab/84BABCB7BABCA6D81/Yifan/sofa_data/generated_scenes &
# done
# wait
# echo "All done"


COLLECT_GENERATOR=sinusoidal COLLECT_DURATION=10 COLLECT_MATRICES=1 COLLECT_SPEED_FACTOR=5.0 \
COLLECT_DIAGNOSTIC_SOLVER=1 COLLECT_VERBOSE=0 COLLECT_DEBUG=1 COLLECT_A_INTERVAL=1 \
python simulation/scenes/collect_data.py \
    --scenes simulation/configs/freespace_full.yaml \
    --scene-idx 1 \
    --output-dir /media/chen-lab/84BABCB7BABCA6D81/Yifan/sofa_data/freespace_full \

# COLLECT_GENERATOR=sinusoidal COLLECT_DURATION=1800 \
# python simulation/scenes/collect_data.py \
#     --scenes simulation/configs/freespace_proximal.yaml \
#     --scene-idx 3 \
#     --output-dir /media/chen-lab/84BABCB7BABCA6D81/Yifan/sofa_data/freespace_proximal &
# wait
# echo "All done"