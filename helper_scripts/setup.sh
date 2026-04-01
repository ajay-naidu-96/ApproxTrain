#!/bin/bash

# Detect OS
OS_NAME=$(uname)

if [ "$OS_NAME" == "Darwin" ]; then
    echo "Detected macOS. Creating 'approxtrain-dev' environment from env.yml..."
    conda env create -f env.yml
    echo ""
    echo "Setup complete. Run 'conda activate approxtrain-dev' to start."
elif [ "$OS_NAME" == "Linux" ]; then
    echo "Detected Linux. Creating 'approxtrain-gpu' environment from env_remote.yml..."
    conda env create -f env_remote.yml
    echo ""
    echo "Setup complete. Run 'conda activate approxtrain-gpu' to start."
else
    echo "Unknown OS: $OS_NAME"
    echo "Please manually create the environment using 'conda env create -f env.yml' (Mac) or 'env_remote.yml' (Linux/CUDA)."
fi
