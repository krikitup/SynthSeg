#!/bin/bash

# Define the base directories
input_dir="data/Brats"
output_dir="results/Brats"

volume_id=185
for type in {1..4}; do
    input_file="${input_dir}/volume_${volume_id}_type${type}.nii.gz"
    output_file="${output_dir}/volume_${volume_id}_type${type}.nii.gz"
    
    
    # Define the mask files
    mask1_file="${input_dir}/volume_${volume_id}_mask1.nii.gz"
    mask2_file="${input_dir}/volume_${volume_id}_mask2.nii.gz"
    mask3_file="${input_dir}/volume_${volume_id}_mask3.nii.gz"
    
    # Check if the mask files exist
    if [ ! -f "$input_file" ] || [ ! -f "$output_file" ] ; then
        echo "Volume ${volume_id} type ${type} do not exist."
        continue
    fi
    # Check if the mask files exist
    if [ ! -f "$mask1_file" ] || [ ! -f "$mask2_file" ] || [ ! -f "$mask3_file" ]; then
        echo "One or more mask files for volume ${volume_id} type ${type} do not exist."
        continue
    fi
   
    
    # Open Freeview with the input, output, and mask files
    freeview -v "$input_file" "$output_file" "$mask1_file" "$mask2_file" "$mask3_file" &
done
