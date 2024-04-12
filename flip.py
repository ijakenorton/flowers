__author__ = "Jake Norton"
__organization__ = "COSC420, University of Otago"
__email__ = "norja159@student.otago.ac.nz"
import numpy as np
import keras
import matplotlib.pyplot as plt
import image_utils
import model_utils
import os
import pickle, gzip
from config import Config, parse_args

from load_oxford_flowers102 import load_oxford_flowers102


def flowers_trained_diffusion():

    train_data, validation_data, test_data, class_names = load_oxford_flowers102(
        imsize=Config.IMAGE_SIZE, fine=Config.FINE
    )

    x_train, y_hat_train, x_test, y_hat_test, x_validation, y_hat_validation = (
        (
            image_utils.augment_data(
                train_data, validation_data, test_data, flip_chance=0.3
            )
        )
        if Config.DATA_AUG
        else (image_utils.normalise_data(train_data, validation_data, test_data))
    )

    #     # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir("saved")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}"):
        os.mkdir(f"{Config.IMAGE_SIZE}")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}"):
        os.mkdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}")

    #     # Specify the names of the save files
    save_name = os.path.join(
        "saved",
        "flowers_encoder%.1e_rdp%.1f_rbn%d_daug%dIMAGE_SIZE%d_aug_flip"
        % (
            Config.REG_WDECAY_BETA,
            Config.REG_DROPOUT_RATE,
            int(Config.REG_BATCH_NORM),
            int(Config.DATA_AUG),
            Config.IMAGE_SIZE,
        ),
    )
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

        net = model_utils.compile_u_ish_net()
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )
        train_info = {}

        x_train, y_hat_train = image_utils.apply_gaussian_noise_with_random_std(
            x_train, (0.0, 2.0)
        )
        x_validation, y_hat_validation = (
            image_utils.apply_gaussian_noise_with_random_std(x_train, (0.0, 2.0))
        )

        train_info = net.fit(
            x_train,
            y_hat_train,
            validation_data=(x_validation, y_hat_validation),
            epochs=Config.EPOCHS,
            shuffle=True,
            callbacks=[model_checkpoint_callback],
        )

        result = [
            y_hat_validation[:16],
            net.predict(x_validation[:16]),
        ]  # Initialize result with initial_noise
        result_array = np.concatenate(result, axis=0)
        # results.append(result)
        image_utils.save_images_to_jpg(
            result_array,
            f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/training_one_fit.jpg",
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
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history["accuracy"], label="accuracy")
        ph.plot(history["val_accuracy"], label="val_accuracy")
        ph.set_xlabel("Epoch")
        ph.set_ylabel("Accuracy")
        ph.set_ylim([0, 1])
        ph.legend(loc="lower right")

    if Config.VERBOSE:
        loss_train, accuracy_train = net.evaluate(x_train, y_hat_train)
        loss_test, accuracy_test = net.evaluate(x_test, y_hat_test)

        print("Train accuracy (: %.2f" % accuracy_train)
        print("Test accuracy  (: %.2f" % accuracy_test)

    image_utils.generate_images(
        net,
        Config.IMAGE_SIZE,
        f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/generated_image_random_200",
    )

    return net


def main():
    parse_args()


if __name__ == "__main__":
    main()
    flowers_trained_diffusion()
