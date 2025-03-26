import numpy as np
import h5py
import os
import tensorflow as tf

model_dir = 'models/'
label_dir = 'data/labels_classes_priors/'

model_name = 'synthseg_2.0.h5'

topological_classes_path = 'synthseg_topological_classes_2.0.npy'
segmentation_labels_path = 'synthseg_segmentation_labels_2.0.npy'
segmentation_names_path = 'synthseg_segmentation_names_2.0.npy'
denoiser_labels_path = 'synthseg_denoiser_labels_2.0.npy'

name_prefix = "Brats"
num_new_classes = 3
unique_num = 100

topological_classes = np.load(os.path.join(label_dir, topological_classes_path))
segmentation_labels = np.load(os.path.join(label_dir, segmentation_labels_path))
segmentation_names = np.load(os.path.join(label_dir, segmentation_names_path))
denoiser_labels = np.load(os.path.join(label_dir, denoiser_labels_path))


updated_topological_classes = np.concatenate((topological_classes, np.ones(num_new_classes)*unique_num))
updated_segmentation_labels = np.concatenate((segmentation_labels, np.arange(unique_num + 1, unique_num + num_new_classes+ 1)))
new_names = np.array(['tumor', 'tumor-core', 'tumor-enhancing'])
updated_segmentation_names = np.concatenate((segmentation_names, new_names))
updated_denoiser_labels = np.concatenate((denoiser_labels, np.ones(num_new_classes)))

# Save the updated files with the prefix "brats_"
np.save(os.path.join(label_dir, f'brats_{topological_classes_path}'), updated_topological_classes)
np.save(os.path.join(label_dir, f'brats_{segmentation_labels_path}'), updated_segmentation_labels)
np.save(os.path.join(label_dir, f'brats_{segmentation_names_path}'), updated_segmentation_names)
np.save(os.path.join(label_dir, f'brats_{denoiser_labels_path}'), updated_denoiser_labels)
np.save(os.path.join(label_dir, f'brats_unique_{segmentation_labels_path}'), np.unique(updated_segmentation_labels))

print("Updated label files saved with prefix 'brats_'.")
