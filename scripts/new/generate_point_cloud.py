import nibabel as nib
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
import os
from skimage import measure

def load_nifti_file(file_path):
    """Load a NIfTI file and return the image data."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def save_class_meshes_from_volume(volume, output_dir="meshes", spacing=(1.0, 1.0, 1.0)):
    """
    Converts a 3D labeled volume into separate .ply meshes for each class.

    Args:
        volume (np.ndarray): 3D array of integers (class labels).
        output_dir (str): Directory to save output .ply files.
        spacing (tuple): Voxel spacing along each axis (z, y, x).
    """
    os.makedirs(output_dir, exist_ok=True)
    class_ids = np.unique(volume)
    
    for class_id in class_ids:
        if class_id == 0:
            continue  # Assuming 0 is background

        binary_mask = (volume == class_id).astype(np.uint8)

        # Extract mesh using marching cubes
        verts, faces, normals, _ = measure.marching_cubes(binary_mask, level=0.5, spacing=spacing)

        # Create and export mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(os.path.join(output_dir, f"class_{class_id}.ply"))

    print(f"Meshes saved to {output_dir}")

if __name__ == "__main__":
    # Input NIfTI file path
    nifti_file = "data/training_label_maps/training_seg_01.nii.gz"  # Replace with your file path
    output_dir = "meshes"  # Replace with desired output file name

    volume = nib.load(nifti_file).get_fdata()
    
    save_class_meshes_from_volume(volume, output_dir=output_dir)
    

    print(f"Point cloud saved to {output_dir}")
