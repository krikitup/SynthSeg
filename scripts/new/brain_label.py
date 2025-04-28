import os
import nibabel as nib
import numpy as np

def process_label_files(directory):
    """
    Process each label/mask file in the given directory, set every non-zero label to 100,
    and save it back with the same name.

    :param directory: Path to the directory containing label/mask files.
    """
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a NIfTI file (e.g., .nii or .nii.gz)
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(directory, filename)
            
            # Load the label/mask file
            print(f"Processing file: {file_path}")
            label_img = nib.load(file_path)
            label_data = label_img.get_fdata()

            # Set every non-zero label to 100
            label_data[label_data != 0] = 100

            # Save the modified file back to the original directory
            modified_img = nib.Nifti1Image(label_data, label_img.affine, label_img.header)
            nib.save(modified_img, file_path)
            print(f"Modified file saved: {file_path}")

# Directory containing the label/mask files
label_directory = "results/training/labels"

# Process the label files
process_label_files(label_directory)