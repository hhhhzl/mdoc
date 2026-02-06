#!/bin/bash
# Run from repo root so mdoc/... paths resolve.
cd "$(dirname "$0")/.." || exit 1

python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e Empty
python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e Random
python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e Narrow
python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e Tennis
python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e Conveyor
python mdoc/baselines/mb_benchmark/run_diffusion_gif.py --e DropRegion

