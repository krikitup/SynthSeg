#!/bin/bash

# Define the base directories
ORIGINAL_DIR="Data/oscar_nii"
SEGMENTED_DIR="Data/output/oscar"

# Create an array of volume suffixes from 'a' to 'm'
VOLUMES=(j)

# Construct the Freeview command
CMD="freeview"

# Add all original volumes
for VOL in "${VOLUMES[@]}"; do
    ORIGINAL_FILE="${ORIGINAL_DIR}/new_oscar_nii${VOL}.nii"
    if [[ -f "$ORIGINAL_FILE" ]]; then
        CMD+=" -v $ORIGINAL_FILE"
    else
        echo "Warning: $ORIGINAL_FILE not found, skipping..."
    fi
done

# Add all segmented volumes with LUT colormap
for VOL in "${VOLUMES[@]}"; do
    SEGMENTED_FILE="${SEGMENTED_DIR}/oscar_nii${VOL}_synthseg.nii"
    if [[ -f "$SEGMENTED_FILE" ]]; then
        CMD+=" -v $SEGMENTED_FILE:colormap=lut"
    else
        echo "Warning: $SEGMENTED_FILE not found, skipping..."
    fi
done

# Run the command
echo "Running: $CMD"
eval "$CMD"
