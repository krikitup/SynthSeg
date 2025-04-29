
def transform_tumor(tumor_mask, gen_img, scale_factor: float, target_center: np.array):
    # Find the center of the tumor in tumor_mask
    tumor_indices = np.argwhere(tumor_mask > 0)
    if tumor_indices.size == 0:
        raise ValueError("No tumor found in mask.")
    tumor_center = np.mean(tumor_indices, axis=0).astype(int)  # (z, y, x)

    # Extract the bounding box of the tumor
    min_idx = tumor_indices.min(axis=0)
    max_idx = tumor_indices.max(axis=0) + 1
    tumor_crop = tumor_mask[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]

    # Scale the tumor crop in 3D
    zoom_factors = [scale_factor, scale_factor, scale_factor]
    scaled_tumor = zoom(tumor_crop, zoom_factors, order=0)  # nearest neighbor

    # Find new center after scaling
    scaled_center = np.array(scaled_tumor.shape) // 2

    # Compute placement indices
    start = target_center - scaled_center
    end = start + np.array(scaled_tumor.shape)

    # Ensure indices are within bounds
    start = np.maximum(start, 0)
    end = np.minimum(end, gen_img.shape)
    insert_slices = tuple(slice(start[i], end[i]) for i in range(3))
    tumor_slices = tuple(slice(0, end[i] - start[i]) for i in range(3))

    # Place the scaled tumor in new_mask
    new_mask = np.zeros_like(gen_img)
    new_mask[insert_slices] = scaled_tumor[tumor_slices]
    return new_mask

def get_z_range(tumor_mask):
    tumor_indices = np.argwhere(tumor_mask > 0)
    if tumor_indices.size == 0:
        raise ValueError("No tumor found in mask.")
    min_idx = tumor_indices.min(axis=0)
    max_idx = tumor_indices.max(axis=0) + 1
    z_start = min_idx[0]
    z_end = max_idx[0]
    return z_start, z_end