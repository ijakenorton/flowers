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

IMAGE_SIZE = 112


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
    x_train = train_data["images"]
    y_hat_train = train_data["labels"]
    x_test = test_data["images"]
    y_hat_test = test_data["labels"]
    x_validation = validation_data["images"]
    y_hat_validation = validation_data["labels"]
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
        "flowers%.1e_rdp%.1f_rbn%d_daug%dIMAGE_SIZE%d"
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
    #     # Show 16 train images with the corresponding labels
    if verbose:
        pass
        # show_methods.show_data_images(
        #     images=x_train_sample, labels=y_hat_train_sample, class_names=class_names
        # )

    n_classes = len(class_names)

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
        # ************************************************
        # * Creating and training a neural network model *
        # ************************************************

        # Create feed-forward network
        net = keras.models.Sequential()

        net.add(
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            )
        )

        net = add_layer(net, 16, 1)

        if reg_batch_norm:
            # Batch norm layer 1
            net.add(keras.layers.BatchNormalization())

        net.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Conv layer 3: 3x3 window, 64 filters

        net = add_layer(net, 32, 2)

        if reg_batch_norm:
            # Batch norm layer 1
            net.add(keras.layers.BatchNormalization())

        net.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        net = add_layer(net, 64, 3)

        net.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        net = add_layer(net, 128, 3)

        if reg_batch_norm:
            # Batch norm layer 2
            net.add(keras.layers.BatchNormalization())

        # Max pool layer 3: 2x2 window (implicit arguments - padding="valid")
        net.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        net = add_layer(net, 128, 3)

        if reg_batch_norm:
            # Batch norm layer 2
            net.add(keras.layers.BatchNormalization())

        net.add(keras.layers.Flatten())

        if reg_wdecay_beta > 0:
            reg_wdecay = keras.regularizers.l2(reg_wdecay_beta)
        else:
            reg_wdecay = None

        if reg_dropout_rate > 0:
            # Dropout layer 1:
            net.add(keras.layers.Dropout(reg_dropout_rate))

        net.add(
            keras.layers.Dense(
                units=2048, activation="relu", kernel_regularizer=reg_wdecay
            )
        )
        net.add(
            keras.layers.Dense(
                units=2048, activation="relu", kernel_regularizer=reg_wdecay
            )
        )
        # Dense layer 3: n_classes neurons
        net.add(keras.layers.Dense(units=n_classes, activation="softmax"))

        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        net.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            # metrics=[keras.metrics.F1Score],
        )
        print(net.summary())

        # Training callback to call on every epoch -- evaluates
        # the model and saves its weights if it performs better
        # (in terms of accuracy) on validation data than any model
        # from previous epochs
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        if data_aug:
            # Read the number of training points - len on numpy array returns the number of rows...
            # should be 50000 for CIFAR10 dataset
            N = len(x_train)
            # Specify the number of points to use for validation
            N_valid = int(N * 0.33)

            # Generate a list of randomly ordered indexes from 0 to N-1
            I = np.random.permutation(N)

            # Select the validation inputs and the corresponding labels
            x_valid = x_train[I[:N_valid]]
            y_hat_valid = y_hat_train[I[:N_valid]]

            # Select the training input and the corresponding labels
            x_train = x_train[I[N_valid:]]
            y_hat_train = y_hat_train[I[N_valid:]]

            # Crate data generator that randomly manipulates images
            datagen = keras.preprocessing.image.ImageDataGenerator(
                zca_epsilon=1e-06,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode="nearest",
                horizontal_flip=True,
            )

            # Configure the data generator for the images in the training sets
            datagen.fit(x_train)

            # Build the data generator
            train_data_aug = datagen.flow(x_train, y_hat_train)

            if verbose:
                for x_batch, y_hat_batch in datagen.flow(
                    x_train_sample, y_hat_train_sample, shuffle=False
                ):
                    # show_methods.show_data_images(
                    #     images=x_batch.astype("uint8"),
                    #     labels=y_hat_batch,
                    #     class_names=class_names,
                    # )

                    break

            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model

            train_info = net.fit(
                train_data_aug,
                validation_data=(x_valid, y_hat_valid),
                epochs=200,
                shuffle=True,
                callbacks=[model_checkpoint_callback],
            )
        else:
            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model
            train_info = net.fit(
                x_train,
                y_hat_train,
                validation_data=(x_validation, y_hat_validation),
                epochs=200,
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
        y_test = np.argmax(y_test, axis=1)
        print(y_test)

        try:
            show_methods.show_data_images(
                images=x_test[:16],
                labels=y_hat_test[:16],
                predictions=y_test,
                class_names=class_names,
            )
        except Exception as e:
            print(f"Error: {e}")

    return net


def add_layer(network, filter_size, count):
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
        # reg_wdecay_beta=0.1,
        reg_dropout_rate=0.4,
        # reg_batch_norm=True,
        data_aug=True,
    )
