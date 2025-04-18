#!/bin/bash

cd ../../
pip install -e .

cd deps/torch_robotics
pip install -e .
cd ../experiment_launcher
pip install -e .
cd ../motion_planning_baselines
pip install -e .
cd ../..

pip install --upgrade --no-cache-dir gdown