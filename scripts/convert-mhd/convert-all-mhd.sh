#!/bin/bash

# Get absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Change to the directory containing docker-compose.yml
cd "$SCRIPT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/data/vdb"

# Iterate over all .mhd files in the data/mhd directory
for mhd_file in "$PROJECT_ROOT/data/mhd"/*.mhd; do
    # Get just the filename without path and extension
    base_name=$(basename "$mhd_file" .mhd)
    
    echo "Converting $base_name..."
    
    # Run the conversion script
    ./convert-mhd.sh "/data/mhd/$base_name.mhd" "/data/vdb/$base_name.vdb"
done

echo "All conversions completed!" 