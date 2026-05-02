#!/bin/bash
set -e

REPO_URL=${1:-""}  # pass your GitHub repo URL as first arg

cd /workspace

# Clone if not already present
if [ ! -d "slm" ]; then
    if [ -z "$REPO_URL" ]; then
        echo "Usage: bash runpod_train.sh <github-repo-url>"
        exit 1
    fi
    git clone "$REPO_URL" slm
fi

cd slm

pip install -r requirements.txt
pip install -e .

python scripts/download_data.py

python scripts/train.py --config configs/model_config.yaml
