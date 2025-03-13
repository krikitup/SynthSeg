#!/bin/bash

# Define the base directories
input_dir="data/Brats"
output_dir="results/Brats"
script_path="scripts/commands/SynthSeg_predict.py"

# Get a list of all volume files for type1
volume_files=($(ls ${input_dir}/volume_*_type1.nii.gz))

# Shuffle the list and pick 5 random volumes
shuffled_files=($(shuf -e "${volume_files[@]}" | head -n 5))

# Loop through the selected volumes and run the prediction script for each type
for volume_file in "${shuffled_files[@]}"; do
    # Extract the volume ID from the filename
    volume_id=$(basename "$volume_file" | cut -d'_' -f2)
    
    for type in {1..4}; do
        input_file="${input_dir}/volume_${volume_id}_type${type}.nii.gz"
        output_file="${output_dir}/volume_${volume_id}_type${type}.nii.gz"
        
        # Run the prediction script
        python $script_path --i $input_file --o $output_file
    done
done