#!/bin/bash

set -e

METHOD="mdoc"

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
    echo "---------------Running MMD experiments---------------"
    python launch_multi_agent_experiment.py --lp 'MPDEnsemble'
else
    echo "---------------Running MDOC experiments---------------"
    python launch_multi_agent_experiment.py --lp 'MPDEnsemble'
fi

echo "Done."