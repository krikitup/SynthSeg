
import os
import numpy as np
import nibabel as nib

for i in range(1, 353):
    # Load the label map
    label_map_path = f"data/Brats_resize/robust_label/volume_{i:03d}_synthseg.nii.gz"
    label_map = nib.load(label_map_path)
    data = label_map.get_fdata()
    tumor_mask_path = f"data/Brats_resize/masks/volume_{i:03d}_mask.nii.gz"
    tumor = nib.load(tumor_mask_path)
    tumor_data = tumor.get_fdata()
    # Merge classes 4 and 5
    data[tumor_data != 0] = tumor_data[tumor_data != 0]

    # Save the modified label map
    new_label_map = nib.Nifti1Image(data, label_map.affine, label_map.header)
    new_label_map_path = f"data/Brats_resize/merged/volume_{i:03d}_mergedmask.nii.gz"
    nib.save(new_label_map, new_label_map_path)
    if i%10 == 0:
        print(f"Processed volume {i:03d}")
print(f"Saved merged label map")