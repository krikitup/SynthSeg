#!/bin/bash

# Define the source and target directories relative to the script's location
SOURCE_DIR="$(dirname "$0")/data/training"
TARGET_DIR="$(dirname "$0")//data/labels"

# Create the target directory if it does not exist
mkdir -p "$TARGET_DIR"

# Move files starting with "labels" from source to target directory
for file in "$SOURCE_DIR"/labels*; 
do
    if [ -e "$file" ]; then
        mv "$file" "$TARGET_DIR"
        echo "Moved: $file -> $TARGET_DIR"
    fi
done