#!/bin/bash

set -e

METHOD="mmd"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method=*) METHOD="${1#*=}" ;;
        --method) shift; METHOD="$1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
cd ../../
cd scripts/inference
# Run Inference
if [[ "$METHOD" == "mmd" ]]; then
    echo "---------------Running MMD---------------"
    python inference_multi_agent.py --method=$METHOD
else
    echo "---------------Running MDOC---------------"
    python inference_multi_agent.py --method=$METHOD
fi

echo "Done."