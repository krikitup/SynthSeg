#!/bin/bash

# Directory containing the mask files
mask_dir="data/Brats_resize/masks"

# Loop through all files matching the naming pattern
for mask_file in "$mask_dir"/*_mask.nii.gz; do
    if [ -f "$mask_file" ]; then
        # Delete the file
        rm "$mask_file"
        echo "Deleted: $mask_file"
    fi
done

echo "All matching mask files have been deleted!"