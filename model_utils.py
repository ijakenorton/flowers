import keras
import matplotlib.pyplot as plt
from config import Config


def compile_u_ish_net() -> keras.models.Sequential:

    net = keras.models.Sequential()
    layers = [2**4, 2**5, 2**6, 2**7]

    net = encoder(net, layers)

    net = decoder(net, layers)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)

    net.compile(
        optimizer=optimizer,
        loss="mae",
        metrics=["accuracy"],
    )
    return net


def encoder(network, filter_list):

    network.add(
        keras.layers.Conv2D(
            filters=filter_list[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
            input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),
        )
    )

    network = add_conv_layer(network, filter_list[0], 1)

    if Config.REG_BATCH_NORM:
        network.add(keras.layers.BatchNormalization())

    network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    for num in filter_list[1:]:
        network = add_conv_layer(network, num, 3)
        if Config.REG_BATCH_NORM:
            network.add(keras.layers.BatchNormalization())
        network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    return network


def decoder(network, filter_list):
    sliced_list = filter_list[1:]
    for num in reversed(sliced_list):
        network = add_conv2d_transpose_layer(network, num)

    network.add(
        keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            activation="sigmoid",
        )
    )
    return network


def add_conv2d_transpose_layer(network, filter_size):
    network.add(
        keras.layers.Conv2DTranspose(
            filters=filter_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation="relu",
            padding="same",
        )
    )
    return network


def add_conv_layer(network, filter_size, count):
    for _ in range(0, count):
        network.add(
            keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            )
        )
    return network
