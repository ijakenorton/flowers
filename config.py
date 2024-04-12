# config.py
import argparse


# config.py


class Config:
    IMAGE_SIZE = 112
    FINE = False
    EPOCHS = 100
    NOISE_RANGE = (0.0, 2.0)
    LOAD_FROM_FILE = False
    VERBOSE = True
    REG_WDECAY_BETA = 0.1
    REG_DROPOUT_RATE = 0.4
    REG_BATCH_NORM = True
    DATA_AUG = False
    LEARNING_RATE = 0.001
    IMAGE_OUTPUT_FOLDER = "images"  # Default output folder

    @classmethod
    def set_image_size(cls, size):
        cls.IMAGE_SIZE = size

    @classmethod
    def set_fine(cls, fine):
        cls.FINE = fine

    @classmethod
    def set_epochs(cls, epochs):
        cls.EPOCHS = epochs

    @classmethod
    def set_noise_range(cls, noise_range):
        cls.NOISE_RANGE = noise_range

    @classmethod
    def set_load_from_file(cls, load):
        cls.LOAD_FROM_FILE = load

    @classmethod
    def set_verbose(cls, verbose):
        cls.VERBOSE = verbose

    @classmethod
    def set_reg_wdecay_beta(cls, beta):
        cls.REG_WDECAY_BETA = beta

    @classmethod
    def set_reg_dropout_rate(cls, rate):
        cls.REG_DROPOUT_RATE = rate

    @classmethod
    def set_reg_batch_norm(cls, batch_norm):
        cls.REG_BATCH_NORM = batch_norm

    @classmethod
    def set_data_aug(cls, aug):
        cls.DATA_AUG = aug

    @classmethod
    def set_learning_rate(cls, rate):
        cls.LEARNING_RATE = rate

    @classmethod
    def set_image_output_folder(cls, folder):
        cls.IMAGE_OUTPUT_FOLDER = folder


def parse_args():
    parser = argparse.ArgumentParser(description="Set configuration parameters.")
    parser.add_argument("--image-size", type=int, help="Set the image size.")
    parser.add_argument("--fine", type=bool, help="Set fine tuning mode.")
    parser.add_argument("--epochs", type=int, help="Set the number of epochs.")
    parser.add_argument(
        "--noise-range-min", type=float, help="Set minimum noise range."
    )
    parser.add_argument(
        "--noise-range-max", type=float, help="Set maximum noise range."
    )
    parser.add_argument("--load-from-file", type=bool, help="Load from file or not.")
    parser.add_argument("--verbose", type=bool, help="Enable verbosity.")
    parser.add_argument("--reg-wdecay-beta", type=float, help="Set weight decay beta.")
    parser.add_argument("--reg-dropout-rate", type=float, help="Set dropout rate.")
    parser.add_argument(
        "--reg-batch-norm", type=bool, help="Enable batch normalization."
    )
    parser.add_argument("--data-aug", type=bool, help="Enable data augmentation.")
    parser.add_argument("--learning-rate", type=float, help="Set the learning rate.")
    parser.add_argument(
        "--image-output-folder", type=str, help="Set the image output folder."
    )

    args = parser.parse_args()

    # Apply configurations only if arguments are provided
    if args.image_size is not None:
        Config.set_image_size(args.image_size)
    if args.fine is not None:
        Config.set_fine(args.fine)
    if args.epochs is not None:
        Config.set_epochs(args.epochs)
    if args.noise_range_min is not None and args.noise_range_max is not None:
        Config.set_noise_range((args.noise_range_min, args.noise_range_max))
    if args.load_from_file is not None:
        Config.set_load_from_file(args.load_from_file)
    if args.verbose is not None:
        Config.set_verbose(args.verbose)
    if args.reg_wdecay_beta is not None:
        Config.set_reg_wdecay_beta(args.reg_wdecay_beta)
    if args.reg_dropout_rate is not None:
        Config.set_reg_dropout_rate(args.reg_dropout_rate)
    if args.reg_batch_norm is not None:
        Config.set_reg_batch_norm(args.reg_batch_norm)
    if args.data_aug is not None:
        Config.set_data_aug(args.data_aug)
    if args.learning_rate is not None:
        Config.set_learning_rate(args.learning_rate)
    if args.image_output_folder is not None:
        Config.set_image_output_folder(args.image_output_folder)

    # Optional print to verify configurations
    print("Configuration has been set.")
