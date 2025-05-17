#!/bin/bash

cd ../../
pip install -r requirements.txt

cd deps/torch_robotics
pip install -e .
cd ../experiment_launcher
pip install -e .
cd ../motion_planning_baselines
pip install -e .
cd ../../

pip install -e .
pip install --upgrade --no-cache-dir gdown