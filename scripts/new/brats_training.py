import os
import numpy as np
import tensorflow as tf
from ext.lab2im import layers
import nibabel as nib
from ext.neuron import models as nrn_models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def combined_loss(y_true, y_pred, alpha=0.5, smooth=1e-6):
    """
    Combined loss function: weighted sum of categorical cross-entropy and Dice loss.

    :param y_true: Ground truth labels (one-hot encoded or binary).
    :param y_pred: Predicted labels (probabilities from the model).
    :param alpha: Weight for the Dice loss (1 - alpha is the weight for categorical cross-entropy).
    :param smooth: Smoothing factor to avoid division by zero in Dice loss.
    :return: Combined loss value.
    """
    # Categorical cross-entropy loss
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Dice loss
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coeff

    # Combined loss
    return alpha * dice_loss + (1 - alpha) * cce_loss

# Define the model path
model_path = "models/brats_synthseg_2.0.h5"  # Replace with the actual path

# Define input parameters
input_shape = [None, None, None, 1]  # 3D input with 1 channel
labels_segmentation = np.arange(36)  # Example label list (update based on your dataset)
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
feat_multiplier = 2
activation = 'elu'
sigma_smoothing = 0
flip_indices = None
gradients = False

# Define the build_model function
def build_model(path_model,
                input_shape,
                labels_segmentation,
                n_levels,
                nb_conv_per_level,
                conv_size,
                unet_feat_count,
                feat_multiplier,
                activation,
                sigma_smoothing,
                flip_indices,
                gradients):
    assert os.path.isfile(path_model), "The provided model path does not exist."

    # Get the number of labels
    n_labels_seg = len(labels_segmentation)

    # Build the UNet
    net = nrn_models.unet(input_shape=input_shape,
                          nb_labels=n_labels_seg,
                          nb_levels=n_levels,
                          nb_conv_per_level=nb_conv_per_level,
                          conv_size=conv_size,
                          nb_features=unet_feat_count,
                          feat_mult=feat_multiplier,
                          activation=activation,
                          batch_norm=-1)
    net.load_weights(path_model, by_name=True, skip_mismatch=True)

    # Smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = tf.keras.models.Model(inputs=net.inputs, outputs=last_tensor)

    return net

# Load the model
model = build_model(model_path,
                    input_shape,
                    labels_segmentation,
                    n_levels,
                    nb_conv_per_level,
                    conv_size,
                    unet_feat_count,
                    feat_multiplier,
                    activation,
                    sigma_smoothing,
                    flip_indices,
                    gradients)

# Print the model summary
model.summary()

# Fine-tune the last two layers
for layer in model.layers[:-2]:
    layer.trainable = False  # Freeze all layers except the last two

# Compile the model with the combined loss
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha=0.5), 
              metrics=['accuracy'])

# Define a custom data generator (replace with your actual data generator)
def data_generator(images, masks, batch_size):
    num_samples = len(images)
    while True:
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = masks[i:i+batch_size]

            # Add channel dimension if necessary
            batch_images = np.expand_dims(batch_images, axis=-1)
            batch_masks = np.expand_dims(batch_masks, axis=-1)

            yield batch_images, batch_masks

# Example: Load your data (replace with actual paths)
image_dir = "data/Brats_resize/images"
mask_dir = "data/Brats_resize/masks"

def load_data(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Load image and mask
        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image (optional)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

images, masks = load_data(image_dir, mask_dir)

# Create data generators
batch_size = 1
train_gen = data_generator(images, masks, batch_size)

# Train the model
steps_per_epoch = len(images) // batch_size
epochs = 50  # Adjust as needed

model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

# Save the fine-tuned model
model.save("models/brats_synthseg_finetuned.h5")