__author__ = "Jake Norton"
__organization__ = "COSC420, University of Otago"
__email__ = "norja159@student.otago.ac.nz"
from keras.src.utils.image_dataset import image_utils
import numpy as np
import keras
import matplotlib.pyplot as plt
import show_methods
import model_utils
import image_utils
import os
import pickle, gzip
from config import Config, parse_args
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from load_oxford_flowers102 import load_oxford_flowers102
from sklearn.utils.class_weight import compute_class_weight


def vgg_classifier():

    if not os.path.isdir("saved"):
        os.mkdir("saved")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}"):
        os.mkdir(f"{Config.IMAGE_SIZE}")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}"):
        os.mkdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}")

    train_data, validation_data, test_data, class_names = load_oxford_flowers102(
        imsize=Config.IMAGE_SIZE, fine=Config.FINE
    )

    x_train, x_test, x_validation, y_hat_train, y_hat_test, y_hat_validation = (
        image_utils.extract_data(train_data, validation_data, test_data)
    )

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_hat_train), y=y_hat_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    n_classes = len(class_names)

    # Specify the names of the save files
    save_name = os.path.join("saved/classifier_", Config.format_config_filename())
    net_save_name = save_name + "_cnn_net.h5"
    checkpoint_save_name = save_name + "_cnn_net.chk"
    history_save_name = save_name + "_cnn_net.hist"

    if Config.LOAD_FROM_FILE and os.path.isfile(net_save_name):
        if Config.VERBOSE:
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

        net = model_utils.compile_vgg_net(n_classes)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )
        train_info = {}

        if Config.DATA_AUG:
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
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode="nearest",
                horizontal_flip=True,
            )

            # Configure the data generator for the images in the training sets
            datagen.fit(x_train)

            # Build the data generator
            train_data_aug = datagen.flow(x_train, y_hat_train)

            train_info = net.fit(
                train_data_aug,
                validation_data=(x_valid, y_hat_valid),
                batch_size=100,
                epochs=Config.EPOCHS,
                shuffle=True,
                callbacks=[model_checkpoint_callback],
                class_weight=class_weight_dict,  # Use the class weights here
            )
        else:
            train_info = net.fit(
                x_train,
                y_hat_train,
                batch_size=100,
                validation_data=(x_validation, y_hat_validation),
                epochs=Config.EPOCHS,
                shuffle=True,
                callbacks=[model_checkpoint_callback],
                class_weight=class_weight_dict,  # Use the class weights here
            )

        print("Loading best save weight from %s..." % checkpoint_save_name)
        net.load_weights(checkpoint_save_name)

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
    if Config.VERBOSE and history != []:
        image_utils.plot_history(history)

    if Config.VERBOSE:
        loss_train, accuracy_train = net.evaluate(x_train, y_hat_train)
        loss_test, accuracy_test = net.evaluate(x_test, y_hat_test)

        print("Train accuracy (: %.2f" % accuracy_train)
        print("Test accuracy  (: %.2f" % accuracy_test)

        y_pred = net.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        conf_matrix = confusion_matrix(y_hat_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        f1 = f1_score(y_hat_test, y_pred, average="weighted")
        print("F1 Score: {:.2f}".format(f1))

        print(classification_report(y_hat_test, y_pred, target_names=class_names))
        try:
            show_methods.show_data_images(
                images=x_test[:16],
                labels=y_hat_test[:16],
                predictions=y_pred,
                class_names=class_names,
            )
        except Exception as e:
            print(f"Error: {e}")

    return net


def main():
    parse_args()
    vgg_classifier()


if __name__ == "__main__":
    main()
