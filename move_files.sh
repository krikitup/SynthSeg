#!/bin/bash

# Define the source and target directories
source_dir="results/training"
target_dir="results/training/images"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Find and move all files starting with "image"
for file in "$source_dir"/image*; do
    if [ -f "$file" ]; then
        mv "$file" "$target_dir"
        echo "Moved: $file -> $target_dir"
    fi
done

echo "All files starting with 'image' have been moved to $target_dir."