import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

# Function to resize volumes
def resize_volume(volume, target_shape):
    # Calculate zoom factors for each axis
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    resized_volume = zoom(volume, zoom_factors, order=1)  # Use linear interpolation (order=1)
    return resized_volume

# Paths to your volumes and masks
volume_directory = 'data/Brats/'  # Replace with your actual path
save_volume_directory = 'data/Brats_resize/images/'  # Replace with your actual path
save_mask_dir = 'data/Brats_resize/masks/'
# Loop over volumes (1 to 369)
for vol_num in range(1, 370):
    # # Create lists to hold the different types of volumes (T1, T2, FLAIR)
    
    vol_num_str = f"{vol_num:03d}"
    # # Load and resize each volume type
    
    volume_filename = os.path.join(volume_directory, f'volume_{vol_num}_type1.nii.gz')
    vol_img = nib.load(volume_filename)
    vol_data = vol_img.get_fdata()
    
    # Resize volume from 160x160x160 to 260x260x260
    resized_vol_data = resize_volume(vol_data, target_shape=(160, 160, 155))
    # Save the combined 4D volume
    fin_vol = nib.Nifti1Image(resized_vol_data, vol_img.affine)
    nib.save(fin_vol, os.path.join(save_volume_directory, f'volume_{vol_num_str}.nii.gz'))
    
    # # Initialize the final mask to zero
    final_mask = np.zeros((160,160,155), dtype=int)

    # Load and resize the 3 masks
    mask_labels = [1, 2, 3]  # Labels for the three different masks
    masks = []
    for mask_type in range(1, 4):
        mask_filename = os.path.join(volume_directory, f'volume_{vol_num}_mask{mask_type}.nii.gz')
        mask_img = nib.load(mask_filename)
        mask_data = mask_img.get_fdata()

        # Resize mask from 160x160x160 to 260x260x260
        resized_mask_data = resize_volume(mask_data, target_shape=(160, 160, 155))

        # # Check for overlap and handle accordingly
        # overlap_mask = (final_mask != 0) & (resized_mask_data != 0)
        # if np.any(overlap_mask):
        #     print(f"Warning: Overlap detected in volume {vol_num} for mask {mask_type}!")
        #     print(f"Voxels in overlap with existing masks: {np.sum(overlap_mask)}")
        
        # Update the final mask: if the mask doesn't overwrite, mark the mask label
        final_mask[resized_mask_data != 0] = 100 +mask_type
    # Save the final mask
    final_mask_img = nib.Nifti1Image(final_mask.astype(np.int32), mask_img.affine)
    nib.save(final_mask_img, os.path.join(save_mask_dir, f'volume_{vol_num}_mask.nii.gz'))

print("Processing completed!")
