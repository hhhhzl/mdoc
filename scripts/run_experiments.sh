#!/bin/bash python

# A* CBS/ ECBS
python scripts/inference/launch_multi_agent_experiment.py \
  --n 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --e EnvEmpty2DRobotPlanarDiskCircle \
  --st 0 \
  --hps CBS ECBS \
  --lps WAStar \
  --rl 1000 \
  --nt 10 \
  --ra
python scripts/inference/launch_multi_agent_experiment.py \
  --n 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --e EnvConveyor2DRobotPlanarBoundary \
  --st 0 \
  --hps CBS ECBS \
  --lps WASTAR \
  --rl 1000 \
  --nt 10 \
  --ra
python scripts/inference/launch_multi_agent_experiment.py \
  --n 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --e EnvConveyor2DRobotPlanarBoundary \
  --st 0 \
  --hps CBS ECBS \
  --lps WASTAR \
  --rl 1000 \
  --nt 10 \
  --ra
python scripts/inference/launch_multi_agent_experiment.py \
  --n 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --e EnvConveyor2DRobotPlanarCircle \
  --st 0 \
  --hps CBS ECBS \
  --lps WASTAR WASTARDATA \
  --rl 1000 \
  --nt 10 \
  --ra


# KCBS/MMD-CBS/MDOC-CBS