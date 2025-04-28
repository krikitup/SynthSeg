#!/bin/bash

# Define input and output directories
input_dir="data/Brats_resize/images"
output_dir="data/Brats_resize/imag"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all files in the input directory
for file in "$input_dir"/*.nii.gz; do
  # Extract the filename without the extension
  filename=$(basename "$file" .nii.gz)

  # Use nibabel to extract the first channel
  python3 - <<EOF
import nibabel as nib
import numpy as np
import os

# Load the 4D volume
input_file = "$file"
output_file = os.path.join("$output_dir", f"${filename}_channel1.nii")

img = nib.load(input_file)
data = img.get_fdata()

# Extract the first channel
channel1 = data[..., 0]

# Save the new volume
new_img = nib.Nifti1Image(channel1, img.affine, img.header)
nib.save(new_img, output_file)
EOF

  echo "Processed: $file -> $output_dir/${filename}_channel1.nii"
done