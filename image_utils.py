import numpy as np
import noise
from PIL import Image
from config import Config


def normalise_data(train_data, validation_data, test_data):
    x_train = train_data["images"] / 255
    y_hat_train = x_train
    x_test = test_data["images"] / 255
    y_hat_test = x_test
    x_validation = validation_data["images"] / 255
    y_hat_validation = x_validation
    y_hat_train = np.squeeze(y_hat_train)
    y_hat_test = np.squeeze(y_hat_test)
    return x_train, y_hat_train, x_test, y_hat_test, x_validation, y_hat_validation


def augment_data(train_data, validation_data, test_data, flip_chance=0.5):
    x_train = train_data["images"] / 255
    x_train = augment_images(x_train, flip_chance)
    y_hat_train = x_train
    x_test = test_data["images"] / 255
    x_test = augment_images(x_test, flip_chance)
    y_hat_test = x_test
    x_validation = validation_data["images"] / 255
    x_validation = augment_images(x_validation, flip_chance)
    y_hat_validation = x_validation
    y_hat_train = np.squeeze(y_hat_train)
    y_hat_test = np.squeeze(y_hat_test)
    return x_train, y_hat_train, x_test, y_hat_test, x_validation, y_hat_validation


def flip_color_channels(images, flip_chance=0.5):
    """Randomly flip color channels of images."""
    # Create a new array of the same shape to hold the flipped images
    flipped_images = np.empty_like(images)

    for i in range(images.shape[0]):
        if np.random.rand() < flip_chance:  # With some probability, flip channels
            channels = np.arange(images.shape[-1])
            np.random.shuffle(channels)
            # Manually assign each channel to ensure correct reshaping
            for j in range(images.shape[-1]):
                flipped_images[i, :, :, j] = images[i, :, :, channels[j]]
        else:
            flipped_images[i] = images[i]

    return flipped_images


def augment_images(images, flip_chance=0.5):
    """Apply noise and randomly flip color channels to augment images."""
    # Add Gaussian noise
    noisy_images = gaussian_noise(images, mean=0.0, std=0.1)

    # Optionally flip color channels
    noisy_flipped_images = flip_color_channels(noisy_images, flip_chance=1)

    # Combine original and augmented images
    augmented_images = np.concatenate([images, noisy_flipped_images], axis=0)
    return augmented_images


def gaussian_noise(images, mean=0.0, std=0.1):
    """Apply Gaussian noise to a batch of images."""
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)  # Ensure pixel values are valid
    return noisy_images


def augment_data_with_noise(x_data):
    """Augment data by adding noisy versions of each image."""
    noisy_x_data = gaussian_noise(x_data, mean=0.0, std=0.2)  # Adjust std as needed
    augmented_x_data = np.concatenate((x_data, noisy_x_data), axis=0)
    return augmented_x_data


def generate_random_image(width, height):
    random_image = (
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).astype("float32")
        / 255
    )
    return random_image


def generate_black_image(width, height):
    """Generate a completely black image."""
    # Create an array filled with zeros
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image.astype("float32") / 255


def generate_white_image(width, height):
    """Generate a completely white image."""
    # Create an array filled with 255s
    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    return white_image.astype("float32") / 255


def generate_red_image(width, height):
    """Generate a completely red image."""
    # Create an array with zeros and set the red channel to 255
    red_image = np.zeros((height, width, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # Red channel is the first channel
    return red_image.astype("float32") / 255


def generate_green_image(width, height):
    """Generate a completely green image."""
    # Create an array with zeros and set the green channel to 255
    green_image = np.zeros((height, width, 3), dtype=np.uint8)
    green_image[:, :, 1] = 255  # Green channel is the second channel
    return green_image.astype("float32") / 255


def generate_blue_image(width, height):
    """Generate a completely blue image."""
    # Create an array with zeros and set the blue channel to 255
    blue_image = np.zeros((height, width, 3), dtype=np.uint8)
    blue_image[:, :, 2] = 255  # Blue channel is the third channel
    return blue_image.astype("float32") / 255


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


def generate_images(net, IMAGE_SIZE, file_path):

    def generate_from_noise(initial_noise_func, type):

        for j in range(0, 2, 1):

            print(f"Generating image from {type} {j}.....")
            initial_noise = initial_noise_func(IMAGE_SIZE, IMAGE_SIZE)
            initial_noise = np.expand_dims(initial_noise, axis=0)
            result = [initial_noise]

            y_test = net.predict(initial_noise)
            for i in range(200, 0, -1):
                y_test = net.predict(y_test, verbose=False)
                result.append(y_test)

            result_array = np.concatenate(result, axis=0)
            save_images_to_jpg(
                result_array,
                f"{file_path}_{j}_{type}.jpg",
            )

    generate_from_noise(generate_random_image, "uniform")
    generate_from_noise(generate_perlin_noise_image, "perlin")
    generate_from_noise(generate_black_image, "black")
    generate_from_noise(generate_red_image, "red")
    generate_from_noise(generate_blue_image, "blue")
    generate_from_noise(generate_green_image, "green")
    generate_from_noise(generate_white_image, "white")
    generate_from_noise(generate_speckle_noise_image, "speckle")
    generate_from_noise(generate_salt_pepper_noise_image, "salt_pepper")


def apply_gaussian_noise_with_random_std(first, std_range=(0.1, 0.5)):
    noisy_first = np.empty_like(first)
    noisy_second = np.empty_like(first)
    for i, image in enumerate(first):
        std1 = np.random.uniform(std_range[0], std_range[1])
        mean = np.random.uniform(0.0, 0.3)
        std2 = std1 + 0.1

        # Apply Gaussian noise to the first image
        noise1 = np.random.normal(mean, std1, image.shape)
        noisy_image1 = np.clip(image + noise1, 0, 1)

        # Apply Gaussian noise to the second image with a slightly higher std
        noise2 = np.random.normal(mean, std2, image.shape)
        noisy_image2 = np.clip(image + noise2, 0, 1)

        # Store the noisy images
        noisy_first[i] = noisy_image1
        noisy_second[i] = noisy_image2

    return noisy_first, noisy_second


def generate_salt_pepper_noise_image(
    width, height, salt_pepper_ratio=0.01, amount=0.05
):
    """Generate an image with salt and pepper noise."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_pepper_ratio))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], :] = 0

    return image.astype("float32") / 255


def generate_perlin_noise_image(width, height, scale=10):
    """Generate a Perlin noise image."""
    image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            image[i][j] = noise.pnoise2(
                i / scale, j / scale, repeatx=width, repeaty=height
            )
    return np.clip(image, 0, 1)


def generate_speckle_noise_image(width, height, mean=0.0, std=0.1):
    """Generate an image with speckle noise."""
    image = np.ones((height, width, 3), dtype="float32")  # Start with a uniform image
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
