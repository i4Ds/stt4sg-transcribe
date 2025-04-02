#!/bin/bash

# Define the directory for the virtual environment
VENV_DIR="venv"

# Create the virtual environment using Python 3.11
python3.11 -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip to the latest version
pip install --upgrade pip

echo "Virtual environment '$VENV_DIR' created and activated using Python 3.11."
