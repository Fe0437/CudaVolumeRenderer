#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./convert-mhd.sh input.mhd output.vdb"
    echo "Example: ./convert-mhd.sh /data/mhd/artfix_small.mhd /data/artfix_small.vdb"
    exit 1
fi

# Disable path conversion in Git Bash
export MSYS_NO_PATHCONV=1

# Get absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create output directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/../../data"

# Change to the directory containing docker-compose.yml
cd "$SCRIPT_DIR"

# Build the image if needed
docker-compose build

# Run the conversion with the exact paths provided
docker-compose run --rm mhd2vdb python3 scripts/convert-mhd/mhd_to_vdb.py "$1" "$2" 