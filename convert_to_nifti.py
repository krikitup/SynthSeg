import os
import nibabel as nib
import numpy as np
from PIL import Image

# Define the input and output directories
input_dir = 'data/NINS_Dataset/Brain Infection'  # Adjust this path if needed
output_path = 'data/nii_images/NINS_Dataset_BI.nii'  # Adjust this path if needed

# Define the target shape for resizing (e.g., 256x256)
target_shape = (256, 256)

# List to hold all image arrays
image_list = []

# Loop through all files in the input directory (not subdirectories)
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Define the input path
        image_path = os.path.join(input_dir, filename)

        # Load the image using PIL and convert to grayscale
        image = Image.open(image_path).convert('L')

        # Resize the image to the target shape
        image_resized = image.resize(target_shape, Image.ANTIALIAS)
        image_np = np.array(image_resized)

        # Append the image array to the list
        image_list.append(image_np)

# Stack all images into a single 3D array
image_stack = np.stack(image_list, axis=0)

# Create a NIfTI image
nifti_image = nib.Nifti1Image(image_stack, affine=np.eye(4))

# Save the NIfTI image
nib.save(nifti_image, output_path)

print(f'Combined images saved to {output_path}')