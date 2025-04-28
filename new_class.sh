#!/bin/bash

# Directories
VOLUME_DIR="data/Brats_resize/images"  # Replace with the path to the volume folder
MASK_DIR="data/Brats_resize/masks"      # Replace with the path to the mask folder
OUTPUT_MASK_DIR="data/Brats_resize/new_mask"  # Replace with the path to save updated masks

# Ensure the output directory exists
mkdir -p "$OUTPUT_MASK_DIR"

# Loop through all volume files
for VOLUME_FILE in "$VOLUME_DIR"/volume_*.nii.gz; do
    # Extract the base name (e.g., volume_001)
    BASE_NAME=$(basename "$VOLUME_FILE" .nii.gz)

    # Construct the corresponding mask file path
    MASK_FILE="$MASK_DIR/${BASE_NAME}_mask.nii.gz"

    # Check if the mask file exists
    if [ ! -f "$MASK_FILE" ]; then
        echo "Mask file not found for volume: $BASE_NAME"
        continue
    fi

    # Construct the output mask file path
    OUTPUT_MASK_FILE="$OUTPUT_MASK_DIR/${BASE_NAME}_mask.nii.gz"

    # Use nibabel in Python to process the volume and mask
    python3 - <<EOF
import nibabel as nib
import numpy as np

# Load volume and mask
volume = nib.load("$VOLUME_FILE").get_fdata()
mask = nib.load("$MASK_FILE").get_fdata()
volume = volume[:,:,:,0]
# Add 100 to mask regions where volume is non-zero but mask is 0
updated_mask = mask.copy()
updated_mask[(volume > 0) & (mask == 0)] += 100

# Save the updated mask
nib.save(nib.Nifti1Image(updated_mask, affine=nib.load("$MASK_FILE").affine), "$OUTPUT_MASK_FILE")
EOF

    echo "Processed and saved updated mask: $OUTPUT_MASK_FILE"
done