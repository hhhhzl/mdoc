#!/bin/bash

set -e

# install unzip
# apt-get update && apt-get install -y unzip

cd ../../
MODEL_SIZE="full"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model=*) MODEL_SIZE="${1#*=}" ;;
        --model) shift; MODEL_SIZE="$1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Download trajectories if they don't exist
if [ ! -d "data_trajectories" ]; then
    echo "Downloading data_trajectories.tar.xz..."
    gdown 1Onw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
    tar -xJvf data_trajectories.tar.xz
    rm -rf data_trajectories.tar.xz
else
    echo "data_trajectories already exists, skipping download."
fi

# Download model based on size if it doesn't exist
if [ ! -d "data_trained_models" ]; then
    if [[ "$MODEL_SIZE" == "small" ]]; then
        echo "Downloading smaller trained model..."
        gdown 1idBod6n8u38skqMwe4PEeAUFR1TiMC8h
        tar -xJvf data_trained_models_small.tar.xz
        mv data_trained_models_small data_trained_models
        rm -rf data_trained_models_small.tar.xz
    else
        echo "Downloading full trained model..."
        gdown 1WO3tpvg-HU0m9RyDvGyfDamo7roBYMud
        tar -xJvf data_trained_models.tar.xz
        rm -rf data_trained_models.tar.xz
    fi
else
    echo "data_trained_models already exists, skipping download."
fi

echo "Done."