__author__ = "Jake Norton"
__organization__ = "COSC420, University of Otago"
__email__ = "norja159@student.otago.ac.nz"
import numpy as np
import keras
import image_utils
import model_utils
import os
import pickle, gzip
from config import Config, parse_args


def flowers_trained_unet(net_save_name, checkpoint_save_name, history_save_name):
    x_train, y_hat_train, x_test, y_hat_test, x_validation, y_hat_validation = (
        image_utils.load_diffusion_data()
    )

    # Create necessary folders
    if not os.path.isdir("saved"):
        os.mkdir("saved")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}"):
        os.mkdir(f"{Config.IMAGE_SIZE}")

    if not os.path.isdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}"):
        os.mkdir(f"{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}")

    history = []
    #     # Specify the names of the save files

    net = model_utils.compile_u_net()
    train_info = {}

    if not Config.ENCODER:

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )

        train_info = net.fit(
            x_train,
            y_hat_train,
            validation_data=(x_validation, y_hat_validation),
            epochs=Config.EPOCHS,
            shuffle=True,
            callbacks=[model_checkpoint_callback],
        )

        # Leaving this commented out as not sure if you want to run the
        # extra generation during training as it takes a LOT longer.
        # Replace with below if you are curious though
        # generate_images_callback = image_utils.GenerateImagesCallback(
        #     net=net,
        #     image_size=Config.IMAGE_SIZE,
        #     file_path=f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/training_",
        #     interval=10,
        # )
        # train_info = net.fit(
        #     x_train,
        #     y_hat_train,
        #     validation_data=(x_validation, y_hat_validation),
        #     epochs=Config.EPOCHS,
        #     shuffle=True,
        #     callbacks=[model_checkpoint_callback, generate_images_callback],
        # )
    else:

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
            epochs=Config.EPOCHS,
            shuffle=True,
            callbacks=[model_checkpoint_callback],
        )

        # outputs an image snapshot during training
        result = [
            y_hat_validation[:32],
            net.predict(x_validation[:32]),
        ]
        result_array = np.concatenate(result, axis=0)
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
    print(history)
    with gzip.open(history_save_name, "w") as f:
        pickle.dump(history, f)

    if Config.VERBOSE:
        predict(net, history, x_train, y_hat_train, x_test, y_hat_test)

    return net


def predict(net, history, x_train, y_hat_train, x_test, y_hat_test):

    if Config.VERBOSE and history != []:
        print("here")
        image_utils.plot_history(history)

    if Config.VERBOSE and Config.ENCODER:
        loss_train, accuracy_train = net.evaluate(x_train, y_hat_train)
        loss_test, accuracy_test = net.evaluate(x_test, y_hat_test)

        print("Train accuracy (: %.2f" % accuracy_train)
        print("Test accuracy  (: %.2f" % accuracy_test)

        y_pred = net.predict(x_test[:16])  # Predict the first 16 images

        # Calculate error metrics
        mean_error, std_deviation = calculate_error_metrics(y_hat_test[:16], y_pred)

        print(f"Mean per-pixel error: {mean_error}")
        print(f"Standard deviation of per-pixel error: {std_deviation}")
        print(f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/before.jpg")

        image_utils.save_images_to_jpg(
            x_test[:16],
            f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/before.jpg",
        )
        image_utils.save_images_to_jpg(
            y_pred[:16], f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/after.jpg"
        )

    else:

        image_utils.generate_images(
            net,
            Config.IMAGE_SIZE,
            f"./{Config.IMAGE_SIZE}/{Config.IMAGE_OUTPUT_FOLDER}/400_generations",
        )


def calculate_error_metrics(true_images, predicted_images):
    # Compute absolute per-pixel errors across the entire batch
    errors = np.abs(true_images - predicted_images)

    # Flatten errors to a single array of errors for simplicity in calculating statistics
    errors = errors.flatten()

    # Calculate mean and standard deviation of the errors
    mean_error = np.mean(errors)
    std_deviation = np.std(errors)

    return mean_error, std_deviation


def main():
    parse_args()
    net_save_name, checkpoint_save_name, history_save_name = model_utils.model_name()

    if Config.LOAD_FROM_FILE and os.path.isfile(net_save_name):
        if Config.VERBOSE:
            print("Loading neural network from %s..." % net_save_name)
        net = keras.models.load_model(net_save_name)

        history = []
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)

        x_train, y_hat_train, x_test, y_hat_test, _, _ = (
            image_utils.load_diffusion_data()
        )
        predict(net, history, x_train, y_hat_train, x_test, y_hat_test)
    else:
        flowers_trained_unet(net_save_name, checkpoint_save_name, history_save_name)


if __name__ == "__main__":
    main()
