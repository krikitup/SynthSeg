import nibabel as nib
import numpy as np
from plyfile import PlyData, PlyElement

def load_nifti_file(file_path):
    """Load a NIfTI file and return the image data."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def create_point_cloud(data):
    """Generate a point cloud for non-zero values in the data."""
    points = np.argwhere(data > 0)  # Get indices of non-zero values
    values = data[data > 0]         # Get corresponding values
    return points, values

def assign_colors(values):
    """Assign colors based on voxel values."""
    max_value = np.max(values)
    normalized_values = (values / max_value * 255).astype(np.uint8)
    colors = np.zeros((len(values), 3), dtype=np.uint8)
    colors[:, 0] = normalized_values  # Red channel
    colors[:, 1] = 255 - normalized_values  # Green channel
    colors[:, 2] = (normalized_values // 2)  # Blue channel
    return colors

def save_as_ply(points, colors, output_file):
    """Save the point cloud as a PLY file."""
    vertices = np.array(
        [(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )
    ply_element = PlyElement.describe(vertices, 'vertex')
    PlyData([ply_element]).write(output_file)

if __name__ == "__main__":
    # Input NIfTI file path
    nifti_file = "data/training_label_maps/training_seg_01.nii.gz"  # Replace with your file path
    output_ply_file = "ply/training_seg_01.ply"  # Replace with desired output file name

    # Load NIfTI data
    data = load_nifti_file(nifti_file)

    # Create point cloud
    points, values = create_point_cloud(data)

    # Assign colors
    colors = assign_colors(values)

    # Save as PLY
    save_as_ply(points, colors, output_ply_file)

    print(f"Point cloud saved to {output_ply_file}")
