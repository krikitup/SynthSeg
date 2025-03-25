import os
import h5py
import numpy as np
import nibabel as nib
from collections import defaultdict

def convert_h5_to_nii(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    volumes = defaultdict(list)  # Store slices by volume number
    
    # Scan directory for .h5 files
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".h5"):
            volume_id, slice_number = file.rsplit("_slice_", 1)  # Extract volume ID and slice number
            file_path = os.path.join(input_dir, file)

            try:
                with h5py.File(file_path, 'r') as h5_file:
                    if 'data' not in h5_file:
                        print(f"Skipping {file}: 'data' key not found")
                        continue
                    
                    data = h5_file['data'][:]
                    volumes[volume_id].append((int(slice_number.split(".")[0]), data))  # Store with slice number

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Convert each volume to .nii.gz
    for volume_id, slices in volumes.items():
        slices.sort(key=lambda x: x[0])  # Sort slices by slice number
        volume_data = np.stack([s[1] for s in slices], axis=-1)  # Reconstruct 3D volume

        nii_image = nib.Nifti1Image(volume_data, affine=np.eye(4))
        output_path = os.path.join(output_dir, f"{volume_id}.nii.gz")
        nib.save(nii_image, output_path)
        print(f"Saved: {output_path}")

# Example usage
convert_h5_to_nii("data/BraTS2020_training_data/content/data", "results/BraTS_nii")
