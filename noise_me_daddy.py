__author__ = "Jake Norton"
__organization__ = "COSC420, University of Otago"
__email__ = "norja159@student.otago.ac.nz"
from keras.src import metrics
import numpy as np
import keras
import matplotlib.pyplot as plt
import show_methods
from PIL import Image
import os
import pickle, gzip

from load_oxford_flowers102 import load_oxford_flowers102

IMAGE_SIZE = 64


def generate_random_image(width, height):
    random_image = (
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).astype("float32")
        / 255
    )
    return random_image


def save_images_to_jpg(images, file_path):
    """
    Save images to a single JPG file, placing images in a new row after every ten images.

    Args:
    - images: Numpy array of images with shape (num_images, width, height, 3).
              RGB values should be normalized between 0 and 1.
    - file_path: File path to save the JPG file.
    """
    num_images = images.shape[0]
    width, height = images.shape[1], images.shape[2]
    images_per_row = 10

    # Calculate the number of rows needed
    num_rows = (num_images + images_per_row - 1) // images_per_row

    # Create a new blank image to paste individual images onto it
    total_width = width * min(num_images, images_per_row)
    total_height = height * num_rows
    combined_image = Image.new("RGB", (total_width, total_height))

    # Paste individual images onto the blank image
    for i in range(num_images):
        row = i // images_per_row
        col = i % images_per_row
        # Convert normalized RGB values to 0-255 range
        img_data = (images[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_data, "RGB")
        combined_image.paste(img, (col * width, row * height))

    # Save the combined image to a JPG file
    combined_image.save(file_path)


def compile(reg_batch_norm) -> keras.models.Sequential:

    net = keras.models.Sequential()
    layers = [2**4, 2**5, 2**6, 2**7]

    net = encoder(net, layers, reg_batch_norm)

    net = decoder(net, layers)
    optimizer = keras.optimizers.AdamW(learning_rate=0.01)

    net.compile(
        optimizer=optimizer,
        loss="mae",
        metrics=["accuracy"],
    )
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


def apply_gaussian_noise_with_random_std(first, std_range=(0.1, 0.5)):
    noisy_first = np.empty_like(first)
    noisy_second = np.empty_like(first)
    for i, image in enumerate(first):
        std1 = np.random.uniform(std_range[0], std_range[1])
        std2 = std1 + 0.1

        # Apply Gaussian noise to the first image
        noise1 = np.random.normal(0, std1, (64, 64, 3))
        noisy_image1 = np.clip(image + noise1, 0, 1)

        # Apply Gaussian noise to the second image with a slightly higher std
        noise2 = np.random.normal(0, std2, (64, 64, 3))
        noisy_image2 = np.clip(image + noise2, 0, 1)

        # Store the noisy images
        noisy_first[i] = noisy_image1
        noisy_second[i] = noisy_image2

    return noisy_first, noisy_second


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

        net = compile(reg_batch_norm)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )
        train_info = {}
        stds = []
        for i in range(0, 20):
            stds.append(i / 10)

        results = []

        for i, std in enumerate(stds):
            if i == len(stds) - 1:
                break

            x_train, y_hat_train = apply_gaussian_noise_with_random_std(
                x_train, (0.0, 2.0)
            )
            x_validation, y_hat_validation = apply_gaussian_noise_with_random_std(
                x_train, (0.0, 2.0)
            )

            train_info = net.fit(
                x_train,
                y_hat_train,
                validation_data=(x_validation, y_hat_validation),
                epochs=1000,
                shuffle=True,
                # callbacks=[model_checkpoint_callback],
            )

            result = [
                y_hat_validation[:1],
                net.predict(x_validation[:1]),
            ]  # Initialize result with initial_noise
            result_array = np.concatenate(result, axis=0)
            # results.append(result)
            save_images_to_jpg(
                result_array, f"./{IMAGE_SIZE}/images/training{stds[i]}.jpg"
            )
            print(f"comparing {stds[i]} with {stds[i + 1]}")

            y_validation = net.predict(x_validation[:16])
            show_methods.show_data_images(images=x_validation[:16], blocking=False)
            show_methods.show_data_images(images=y_validation, blocking=False)

        # Load the weights of the best model

        # result_array = np.concatenate(result_array, axis=0)
        # save_images_to_jpg(result_array, f"./images/training_all.jpg")
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

    try:
        initial_noise = generate_random_image(IMAGE_SIZE, IMAGE_SIZE)
        initial_noise = np.expand_dims(initial_noise, axis=0)
        result = [initial_noise]  # Initialize result with initial_noise

        y_test = net.predict(initial_noise)
        for i in range(100, 0, -1):
            # show_methods.show_data_images(images=y_test, blocking=False)
            y_test = net.predict(y_test)
            result.append(y_test)  # Append y_test to the result list

        # show_methods.show_data_images(images=y_test, blocking=True)

        # After the loop, concatenate all elements of result along the first axis
        result_array = np.concatenate(result, axis=0)
        save_images_to_jpg(result_array, f".{IMAGE_SIZE}/images/image.jpg")

        print("everywhere")
    except Exception as e:
        print(f"Error: {e}")

    return net


if __name__ == "__main__":
    flowers_trained_cnn(
        load_from_file=True,
        verbose=True,
        reg_wdecay_beta=0.1,
        reg_dropout_rate=0.4,
        reg_batch_norm=True,
        # data_aug=True,
    )
