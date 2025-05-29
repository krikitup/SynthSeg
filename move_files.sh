#!/bin/bash

# Usage: ./move_volumes.sh <source_folder> <destination_folder> <start_num> <end_num>

# Input arguments
SOURCE_FOLDER=$1
DEST_FOLDER=$2
START_NUM=$3
END_NUM=$4

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <source_folder> <destination_folder> <start_num> <end_num>"
    exit 1
fi

# Create the destination folder if it doesn't exist
if [ ! -d "$DEST_FOLDER" ]; then
    mkdir -p "$DEST_FOLDER"
    echo "Created destination folder: $DEST_FOLDER"
fi

# Loop through the specified range of numbers
for ((i=START_NUM; i<=END_NUM; i++)); do
    FILE_NAME="volume_${i}_mask.nii.gz"
    SOURCE_FILE="${SOURCE_FOLDER}/${FILE_NAME}"

    # Check if the file exists in the source folder
    if [ -f "$SOURCE_FILE" ]; then
        mv "$SOURCE_FILE" "$DEST_FOLDER"
        echo "Moved $SOURCE_FILE to $DEST_FOLDER"
    else
        echo "File $SOURCE_FILE does not exist, skipping..."
    fi
done

echo "File transfer completed."