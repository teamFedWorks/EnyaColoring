import glob
import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GroupNormalization,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    Multiply,
    Softmax,
    UpSampling2D,
    Reshape,
)

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Training function using fit
from tqdm import tqdm

InstanceNormalization = lambda: GroupNormalization(groups=-1)


class MultLayer(Layer):
    def call(self, a, b, transpose_b=False):
        return tf.matmul(a, b, transpose_b=transpose_b)


def AB(input_tensor):
    channels = input_tensor.shape[-1]
    # Generate attention map
    f = Conv2D(channels // 8, (1, 1), padding="same")(input_tensor)  # Downscale
    g = Conv2D(channels // 8, (1, 1), padding="same")(input_tensor)  # Downscale
    h = Conv2D(channels // 2, (1, 1), padding="same")(
        input_tensor
    )  # Intermediate representation

    # Compute attention map
    attention_map = Softmax()(MultLayer()(f, g, transpose_b=True))

    # Compute attention features
    attention_features = MultLayer()(attention_map, h)

    # Align channel dimensions
    attention_features = Conv2D(channels, (1, 1), padding="same")(attention_features)

    # Combine input with attention features
    return Multiply()([input_tensor, attention_features])


def SEB(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    se = Reshape((1, 1, channels))(se)
    return Multiply()([input_tensor, se])


def ColoringModel(input_shape=(256, 256, 12)):
    custom_input = Input(shape=input_shape)
    L = custom_input[..., 0:1]  # L channel
    DA = custom_input[..., 1:2]  # Depth mask
    DB = custom_input[..., 2:3]  # Contour mask


    extra_channels = custom_input[..., 1:]  

    combined_inputs = Concatenate(axis=-1)([L, extra_channels])

    def reconstruct_rgb(L):
        """
        Convert normalized L-channel grayscale to pseudo-RGB for ResNet preprocessing.
        The input L is expected to be in range [0, 1].
        """
        L_scaled = L * 255.0
        grayscale = tf.concat([L_scaled, L_scaled, L_scaled], axis=-1)
        return grayscale

    # Reconstruct RGB images for ResNet preprocessing
    rgb_images = Lambda(reconstruct_rgb)(L)


    # Use ResNet50 backbone for feature extraction

    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=rgb_images
    )

    # Freeze ResNet layers except for the last few layers
    for layer in base_model.layers[:-8]:
        layer.trainable = False
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    # Resize to match helper function
    def RM(input_tensor, reference_tensor):
        ref_shape = reference_tensor.shape[1:3]
        return Lambda(lambda x: tf.image.resize(x, ref_shape))(input_tensor)

    # Encoder Features
    c1 = base_model.get_layer("conv1_relu").output
    c2 = base_model.get_layer("conv2_block3_out").output
    c3 = base_model.get_layer("conv3_block4_out").output
    c4 = base_model.get_layer("conv4_block6_out").output
    combined_c1 = Concatenate()([c1, RM(combined_inputs[..., :], c1)])
    combined_c1 = AB(combined_c1)  # Apply Attention Block

    combined_c2 = Concatenate()([c2, RM(combined_inputs[..., :], c2)])
    combined_c2 = SEB(combined_c2)  # Apply Squeeze-Excite Block

    combined_c3 = Concatenate()([c3, RM(combined_inputs[..., :], c3)])
    combined_c3 = AB(combined_c3)  # Apply Attention Block

    combined_c4 = Concatenate()([c4, RM(combined_inputs[..., :], c4)])
    combined_c4 = SEB(combined_c4)  # Apply Squeeze-Excite Block
    print(combined_c1.shape,combined_c2.shape,combined_c3.shape,combined_c4.shape)
    # Decoder Block
    def DB(skip_connection, upsample_input, filters, L_input):
        x = UpSampling2D()(upsample_input)
        resized_L = RM(L_input, x)  # Resize L to match the current tensor
        x = Concatenate()([x, skip_connection, resized_L])
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = InstanceNormalization()(x)
        x = AB(x)
        x = SEB(x)
        return x

    # Decoders for each output channel
    def CD(output_name, filters, skip_connections, activation_fn="sigmoid"):
        """
        Decoder block for a single output channel.

        Args:
            output_name (str): Name of the output channel.
            filters (list): List of filter sizes for each stage of the decoder.
            skip_connections (list): List of skip connections from encoder.
            activation_fn (str): Activation function ('tanh' or 'sigmoid').

        Returns:
            Tensor: Output tensor for the specified channel.
        """
        d1 = DB(skip_connections[2], skip_connections[3], filters[0], combined_inputs)
        d2 = DB(skip_connections[1], d1, filters[1], combined_inputs)
        d3 = DB(skip_connections[0], d2, filters[2], combined_inputs)
        d4 = UpSampling2D()(d3)
        d4 = Conv2D(64, (3, 3), padding="same", activation="relu")(d4)
        d4 = InstanceNormalization()(d4)
        output = Conv2D(2, (1, 1), activation=activation_fn, name=output_name)(d4)
        return output

    # Filters for each decoding stage
    filters = [512, 256, 128]

    # Separate decoders for each output channel
    combined_output = CD(
        "two_channels",
        filters,
        [combined_c1, combined_c2, combined_c3, combined_c4],
        "sigmoid",
    )


    # Model Definition
    model = Model(inputs=custom_input, outputs=combined_output, name="ColoringModel")
    return model
