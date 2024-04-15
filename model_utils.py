import keras
from keras import layers
import matplotlib.pyplot as plt
from config import Config
import os


def compile_vgg_net(n_classes) -> keras.models.Model:
    input_layer = keras.Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    layers = [2**4, 2**5, 2**6, 2**7]

    encoder_output = encoder(input_layer, layers)

    net_output = classifier_decoder(encoder_output, n_classes)

    net = keras.models.Model(inputs=input_layer, outputs=net_output)

    optimizer = keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE)

    # Compile the model
    net.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return net


def compile_u_net() -> keras.models.Model:
    input_layer = keras.Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    layers = [2**4, 2**5, 2**6, 2**7]

    if Config.SKIPS:
        encoder_output, skip_connections = encoder(input_layer, layers)
    else:
        encoder_output = encoder(input_layer, layers)

    if Config.SKIPS:
        net_output = decoder(encoder_output, layers, skip_connections)
    else:
        net_output = decoder(encoder_output, layers)

    net = keras.models.Model(inputs=input_layer, outputs=net_output)

    optimizer = keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE)

    net.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["accuracy"],
    )
    return net


def encoder(inputs, filter_list):
    skip_connections = []
    x = keras.layers.Conv2D(
        filters=filter_list[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        padding="same",
    )(inputs)

    if Config.SKIPS:
        skip_connections.append(x)

    x = add_conv_layer(x, filter_list[0], 1)

    if Config.REG_BATCH_NORM:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for num in filter_list[1:]:
        x = add_conv_layer(x, num, 3)
        if Config.REG_BATCH_NORM:
            x = keras.layers.BatchNormalization()(x)
        if Config.SKIPS:
            skip_connections.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    if Config.SKIPS:
        return x, skip_connections
    else:
        return x


def decoder(encoded, filter_list, skip_connections=None):
    x = encoded
    if Config.SKIPS:
        skip_connections = skip_connections[::-1]

    for i, num in enumerate(reversed(filter_list[1:])):
        x = add_conv2d_transpose_layer(x, num)
        if Config.SKIPS:
            x = layers.Concatenate()([x, skip_connections[i]])

    x = layers.Conv2DTranspose(
        filters=3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="sigmoid",
    )(x)
    return x


# Vgg net style decoder
def classifier_decoder(encoded, n_classes):
    x = encoded
    if Config.REG_BATCH_NORM:
        x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = layers.Flatten()(x)

    if Config.REG_WDECAY_BETA > 0:
        reg_wdecay = keras.regularizers.l2(Config.REG_WDECAY_BETA)
    else:
        reg_wdecay = None

    x = layers.Dense(units=100, activation="relu", kernel_regularizer=reg_wdecay)(x)
    x = layers.Dense(units=512, activation="relu", kernel_regularizer=reg_wdecay)(x)
    x = layers.Dense(units=n_classes, activation="softmax")(x)

    return x


def add_conv_layer(input_tensor, filter_size, count):
    x = input_tensor
    for _ in range(count):
        x = layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )(x)
    return x


def add_conv2d_transpose_layer(input_tensor, filter_size):
    return layers.Conv2DTranspose(
        filters=filter_size,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation="relu",
        padding="same",
    )(input_tensor)


def model_name():
    save_name = os.path.join("saved", Config.format_config_filename())
    net_save_name = save_name + "_cnn_net.h5"
    checkpoint_save_name = save_name + "_cnn_net.chk"
    history_save_name = save_name + "_cnn_net.hist"
    return net_save_name, checkpoint_save_name, history_save_name
