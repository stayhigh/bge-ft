#!/bin/bash

# Define the paths from the PATH environment variable
IFS=':' read -ra PATH_ARRAY <<< "$PATH"

# Loop through each path and check if nvidia-smi exists
for path in "${PATH_ARRAY[@]}"; do
    if [ -x "$path/nvidia-smi" ]; then
        echo "Found 'nvidia-smi' in $path"
    fi
done

