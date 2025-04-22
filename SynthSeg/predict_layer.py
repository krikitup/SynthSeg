"""
This script is similar to `predict_synthseg.py` but gathers outputs at intermediate layers of the U-Net.
"""

# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from ext.lab2im import utils, layers
from ext.neuron import models as nrn_models


def build_model_with_intermediate_outputs(path_model_segmentation, labels_segmentation):
    """
    Builds the U-Net model and includes intermediate outputs after each block.
    """
    assert os.path.isfile(path_model_segmentation), "The provided model path does not exist."

    # Get the number of labels
    n_labels_seg = len(labels_segmentation)

    # Build the U-Net model
    net = nrn_models.unet(
        input_shape=[None, None, None, 1],
        nb_labels=n_labels_seg,
        nb_levels=5,
        nb_conv_per_level=2,
        conv_size=3,
        nb_features=24,
        feat_mult=2,
        activation='elu',
        batch_norm=-1,
        name='unet'
    )

    # Load pre-trained weights
    net.load_weights(path_model_segmentation, by_name=True)

    # Extract intermediate outputs
    intermediate_outputs = []
    for layer in net.layers:
        if 'conv' in layer.name or 'pool' in layer.name or 'up' in layer.name:
            intermediate_outputs.append(layer.output)

    # Create a new model with intermediate outputs
    intermediate_model = Model(inputs=net.input, outputs=intermediate_outputs)

    return intermediate_model


def predict_with_intermediate_outputs(path_images, path_model_segmentation, labels_segmentation, output_dir):
    """
    Predicts segmentation and gathers intermediate outputs for visualization.
    """
    # Load the model with intermediate outputs
    model = build_model_with_intermediate_outputs(path_model_segmentation, labels_segmentation)

    # Prepare input images
    image_paths = utils.list_images_in_folder(path_images)
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        # Load and preprocess the image
        image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(image_path, ct=False)

        # Predict intermediate outputs
        intermediate_outputs = model.predict(image)

        # Save intermediate outputs
        base_name = os.path.basename(image_path).replace('.nii.gz', '')
        for idx, feature_map in enumerate(intermediate_outputs):
            output_path = os.path.join(output_dir, f"{base_name}_layer_{idx + 1}.npy")
            np.save(output_path, feature_map)
            print(f"Saved intermediate output for layer {idx + 1} to {output_path}")

        print(f"Processed {image_path}")


def preprocess(path_image, ct, target_res=1., n_levels=5, crop=None, min_pad=None):
    """
    Preprocesses the input image for prediction.
    """
    # Read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
    if n_dims < 3:
        raise Exception('Input should have 3 dimensions.')

    # Resample image if necessary
    target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
    if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
        im_res = target_res
        im, aff = layers.resample_volume(im, aff, im_res)

    # Align image
    im = layers.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)

    # Normalize image
    im = layers.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # Pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    im, pad_idx = layers.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # Add batch and channel axes
    im = utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, input_shape, pad_idx, None