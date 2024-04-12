__author__ = "Jake Norton"
__organization__ = "COSC420, University of Otago"
__email__ = "norja159@student.otago.ac.nz"
from keras.src import metrics
import numpy as np
import keras
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip

from load_oxford_flowers102 import load_oxford_flowers102

IMAGE_SIZE = 128


def flowers_trained_cnn(
    load_from_file=False,
    verbose=True,
    reg_wdecay_beta=0.0,
    reg_dropout_rate=0.0,
    reg_batch_norm=False,
    data_aug=False,
):

    train_data, validation_data, test_data, class_names = load_oxford_flowers102(
        imsize=IMAGE_SIZE, fine=False
    )
    x_train = train_data["images"] / 255

    y_hat_train = x_train
    x_test = test_data["images"] / 255
    y_hat_test = x_test
    x_validation = validation_data["images"] / 255
    y_hat_validation = x_validation
    #     # Data augmentation does not like shape (N,1) for labels, it must
    #     # be shape (N,)...and the squeeze function removes dimensions of size 1
    y_hat_train = np.squeeze(y_hat_train)
    y_hat_test = np.squeeze(y_hat_test)

    #     # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir("saved")

    #     # Specify the names of the save files
    save_name = os.path.join(
        "saved",
        "flowers_encoder%.1e_rdp%.1f_rbn%d_daug%dIMAGE_SIZE%d"
        % (
            reg_wdecay_beta,
            reg_dropout_rate,
            int(reg_batch_norm),
            int(data_aug),
            IMAGE_SIZE,
        ),
    )
    net_save_name = save_name + "_cnn_net.h5"
    checkpoint_save_name = save_name + "_cnn_net.chk"
    history_save_name = save_name + "_cnn_net.hist"

    x_train_sample = x_train[:16]

    y_hat_train_sample = y_hat_train[:16]
    # if verbose:
    #     show_methods.show_data_images(
    #         images=x_train_sample,
    #         labels=y_hat_train_sample,
    #     )

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        net = keras.models.load_model(net_save_name)

        # Load the training history - since it should have been created right after
        # saving the model
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            history = []
    else:

        net = keras.models.Sequential()
        layers = [2**5, 2**6, 2**7, 2**8]

        net = encoder(net, layers, reg_batch_norm)

        net.summary()
        net = decoder(net, layers)

        net.summary()
        net.compile(
            optimizer="adam",
            loss="mae",
            metrics=["accuracy"],
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        train_info = net.fit(
            x_train,
            y_hat_train,
            validation_data=(x_validation, y_hat_validation),
            epochs=50,
            shuffle=True,
            callbacks=[model_checkpoint_callback],
        )

        # Load the weights of the best model
        print("Loading best save weight from %s..." % checkpoint_save_name)
        net.load_weights(checkpoint_save_name)

        # Save the entire model to file
        print("Saving neural network to %s..." % net_save_name)
        net.save(net_save_name)

        # Save training history to file
        history = train_info.history
        with gzip.open(history_save_name, "w") as f:
            pickle.dump(history, f)

    # *********************************************************
    # * Training history *
    # *********************************************************

    # Plot training and validation accuracy over the course of training
    if verbose and history != []:
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history["accuracy"], label="accuracy")
        ph.plot(history["val_accuracy"], label="val_accuracy")
        ph.set_xlabel("Epoch")
        ph.set_ylabel("Accuracy")
        ph.set_ylim([0, 1])
        ph.legend(loc="lower right")

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************

    if verbose:
        loss_train, accuracy_train = net.evaluate(x_train, y_hat_train)
        loss_test, accuracy_test = net.evaluate(x_test, y_hat_test)

        print("Train accuracy (: %.2f" % accuracy_train)
        print("Test accuracy  (: %.2f" % accuracy_test)

        # Compute output for 16 test images
        y_test = net.predict(x_test[:16])

        try:
            show_methods.show_data_images(
                images=x_test[:16],
                blocking=False,
            )

            show_methods.show_data_images(
                images=y_test,
                blocking=True,
            )
        except Exception as e:
            print(f"Error: {e}")

    return net


def encoder(network, filter_list, reg_batch_norm):

    network.add(
        keras.layers.Conv2D(
            filters=filter_list[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    )

    network = add_conv_layer(network, filter_list[0], 1)

    if reg_batch_norm:
        network.add(keras.layers.BatchNormalization())

    network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    for num in filter_list[1:]:
        network = add_conv_layer(network, num, 3)
        if reg_batch_norm:
            network.add(keras.layers.BatchNormalization())
        network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    return network


def decoder(network, filter_list):
    sliced_list = filter_list[1:]
    print(sliced_list)
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


if __name__ == "__main__":
    flowers_trained_cnn(
        load_from_file=False,
        verbose=True,
        reg_wdecay_beta=0.1,
        reg_dropout_rate=0.4,
        reg_batch_norm=True,
        # data_aug=True,
    )