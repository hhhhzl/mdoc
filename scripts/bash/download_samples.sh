#!/bin/bash

set -e

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

# Always download trajectories
echo "Downloading data_trajectories.tar.xz..."
gdown --id 10nw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
tar -xJvf data_trajectories.tar.xz

# Download model based on size
if [[ "$MODEL_SIZE" == "small" ]]; then
    echo "Downloading smaller trained model..."
    gdown --id 1idBod6n8u38skqMwe4PEeAUFR1TiMC8h
    tar -xJvf data_trained_models_small.tar.xz
    mv data_trained_models_small data_trained_models
else
    echo "Downloading full trained model..."
    gdown --id 1W03tpvg-HU0m9RyDvGyfDamo7roBYMud
    tar -xJvf data_trained_models.tar.xz
fi

echo "Done."