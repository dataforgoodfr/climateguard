#!/bin/bash
# This script is used to install the environment on a linux computer or server
# Should work on macOS as well. 
# Do not run as root, when logging into a server run `su ubuntu` first.

set -e

if ! command -v uv &> /dev/null
then
    echo "Installing uv package manager"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

uv sync
source ./.venv/bin/activate