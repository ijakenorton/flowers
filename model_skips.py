import keras
import matplotlib.pyplot as plt
from config import Config


def compile_u_ish_net(use_skip_connections=False) -> keras.models.Model:
    input_layer = keras.Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    layers = [2**4, 2**5, 2**6, 2**7]

    # Build the encoder
    if use_skip_connections:
        encoder_output, skip_connections = encoder(
            input_layer, layers, use_skip_connections=True
        )
    else:
        encoder_output = encoder(input_layer, layers)

    # Build the decoder, optionally using skip connections
    if use_skip_connections:
        net_output = decoder(encoder_output, layers, skip_connections)
    else:
        net_output = decoder(encoder_output, layers)

    # Create the model
    net = keras.models.Model(inputs=input_layer, outputs=net_output)

    # Set up the optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE)

    # Compile the model
    net.compile(
        optimizer=optimizer,
        loss="mae",
        metrics=["accuracy"],
    )
    return net


def encoder(network, filter_list, use_skip_connections=False):
    skip_connections = []

    x = keras.layers.Conv2D(
        filters=filter_list[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
        input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
    )(network)

    if use_skip_connections:
        skip_connections.append(x)

    x = add_conv_layer(x, filter_list[0], 1)

    if Config.REG_BATCH_NORM:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for num in filter_list[1:]:
        x = add_conv_layer(x, num, 3)
        if Config.REG_BATCH_NORM:
            x = keras.layers.BatchNormalization()(x)
        if use_skip_connections:
            skip_connections.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x, skip_connections if use_skip_connections else x


def decoder(encoded, filter_list, skip_connections=None):
    x = encoded
    # If skip connections are used, they need to be in reverse order
    if skip_connections:
        skip_connections = skip_connections[::-1]

    for i, num in enumerate(reversed(filter_list[1:])):
        x = add_conv2d_transpose_layer(x, num)
        if skip_connections:
            # Concatenate skip connection with the current layer
            x = keras.layers.Concatenate()([x, skip_connections[i]])

    x = keras.layers.Conv2DTranspose(
        filters=3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="sigmoid",
    )(x)
    return x


def add_conv_layer(input_tensor, filter_size, count):
    x = input_tensor
    for _ in range(count):
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
    return x


def add_conv2d_transpose_layer(input_tensor, filter_size):
    x = keras.layers.Conv2DTranspose(
        filters=filter_size,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation="relu",
        padding="same",
    )(input_tensor)
    return x
