import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label
import trimesh
import openfbx as ofbx  # Import openfbx

def create_fbx_scene(mesh, color, transparency):
    """
    Create an FBX scene with a specific mesh, color, and transparency.

    :param mesh: Trimesh object to export.
    :param color: Tuple of RGB values (0-1).
    :param transparency: Transparency value (0-1, where 0 is opaque and 1 is fully transparent).
    :return: FBX scene object.
    """
    # Create an FBX scene
    scene = ofbx.Scene()

    # Add a mesh node
    node = scene.create_mesh_node("MeshNode")

    # Add vertices to the mesh
    for vertex in mesh.vertices:
        node.add_vertex(vertex)

    # Add faces to the mesh
    for face in mesh.faces:
        node.add_face(face)

    # Set material properties (color and transparency)
    material = scene.create_material("Material")
    material.set_diffuse_color(color)
    material.set_transparency(transparency)
    node.set_material(material)

    # Add the node to the scene
    scene.add_node(node)

    return scene


def nii_to_fbx(input_file, output_dir):
    """
    Convert a .nii file into .fbx meshes for each class with different colors and transparency.

    :param input_file: Path to the .nii file.
    :param output_dir: Directory to save the output meshes.
    """
    # Load the .nii file
    nii = nib.load(input_file)
    data = nii.get_fdata()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique classes in the data (excluding background class 0)
    unique_classes = np.unique(data)
    unique_classes = unique_classes[unique_classes != 0]  # Exclude background (class 0)

    print(f"Found classes: {unique_classes}")

    # Define colors and transparency for each class
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
    ]
    transparency = 0.5  # Set transparency (0 = opaque, 1 = fully transparent)

    # Iterate over each class and generate a mesh
    for i, class_value in enumerate(unique_classes):
        print(f"Processing class: {class_value}")

        # Create a binary mask for the current class
        binary_mask = (data == class_value).astype(np.uint8)

        # Label connected components (optional, if you want separate meshes for disconnected regions)
        labeled_mask, _ = label(binary_mask)

        # Extract the mesh for the current class
        verts, faces, _, _ = trimesh.voxel.ops.matrix_to_marching_cubes(labeled_mask)

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Assign a color to the class (loop through colors if there are more classes than colors)
        color = colors[i % len(colors)]

        # Create an FBX scene
        scene = create_fbx_scene(mesh, color, transparency)

        # Save the FBX file
        fbx_file = os.path.join(output_dir, f"class_{int(class_value)}.fbx")
        with open(fbx_file, "wb") as f:
            f.write(scene.to_bytes())

        print(f"Saved FBX file: {fbx_file}")

    print("All classes processed successfully.")

# Input .nii file and output directory
input_nii_file = "path/to/your/file.nii"
output_mesh_dir = "path/to/output/directory"

# Convert the .nii file to FBX meshes
nii_to_fbx(input_nii_file, output_mesh_dir)