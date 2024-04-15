# config.py
import argparse


class Config:
    IMAGE_SIZE = 112
    FINE = False
    EPOCHS = 100
    NOISE_RANGE = (0.0, 2.0)
    LOAD_FROM_FILE = False
    VERBOSE = True
    ENCODER = False
    REG_WDECAY_BETA = 0.1
    REG_DROPOUT_RATE = 0.4
    REG_BATCH_NORM = True
    DATA_AUG = False
    FLIP_HORIZONTAL = False
    SKIPS = False
    FLIP_COLOUR = False
    LEARNING_RATE = 0.01
    IMAGE_OUTPUT_FOLDER = "images"

    @classmethod
    def format_config_filename(cls):
        attributes = []

        abbreviations = {
            "IMAGE_SIZE": "IS",
            "FINE": "FN",
            "EPOCHS": "EP",
            "NOISE_RANGE": "NR",
            "VERBOSE": "VB",
            "ENCODER": "ENC",
            "REG_WDECAY_BETA": "RWB",
            "REG_DROPOUT_RATE": "RDR",
            "REG_BATCH_NORM": "RBN",
            "DATA_AUG": "DA",
            "FLIP_HORIZONTAL": "FH",
            "SKIPS": "SK",
            "FLIP_COLOUR": "FC",
            "LEARNING_RATE": "LR",
            "IMAGE_OUTPUT_FOLDER": "IOF",
        }

        for attr, value in cls.__dict__.items():
            if (
                attr in abbreviations
                and not attr.startswith("__")
                and not callable(getattr(cls, attr))
            ):
                key = abbreviations[attr]
                if isinstance(value, bool):
                    value = int(value)
                    attributes.append(f"{key}={value}d")
                elif isinstance(value, (int, float)):
                    spec = "f" if isinstance(value, float) else "d"
                    attributes.append(f"{key}={value}{spec}")
                elif isinstance(value, str):
                    attributes.append(f"{key}={value}s")
                elif isinstance(value, tuple):
                    tuple_str = ",".join(map(str, value))
                    attributes.append(f"{key}=({tuple_str})s")

        return "_".join(attributes)

    @classmethod
    def parse_config_filename(cls, config_str):
        # Reverse the abbreviations dictionary from the given function
        abbreviations_reversed = {
            "IS": "--image-size",
            "FN": "--fine",
            "EP": "--epochs",
            "NR": "--noise-range",
            "VB": "--verbose",
            "ENC": "--encoder",
            "RWB": "--reg-wdecay-beta",
            "RDR": "--reg-dropout-rate",
            "RBN": "--reg-batch-norm",
            "DA": "--data-aug",
            "FH": "--flip-horizontal",
            "SK": "--skips",
            "FC": "--flip-colour",
            "LR": "--learning-rate",
            "IOF": "--image-output-folder",
        }

        # Split the configuration string into key-value pairs
        attributes = config_str.split("_")
        reversed_attrs = {}

        # Replace abbreviated keys with full keys
        for attr in attributes:
            if "=" in attr:
                key, value = attr.split("=")
                if key in abbreviations_reversed:
                    original_key = abbreviations_reversed[
                        key
                    ]  # Get the original attribute name
                    if value.endswith("d"):
                        value = bool(value[:-1])  # Remove the 'd' and convert to int
                    elif value.endswith("f"):
                        value = float(value[:-1])
                    elif value.endswith("s"):
                        if value.startswith("(") and value.endswith(")s"):
                            value = value[1:-2].split(
                                ","
                            )  # Remove the '(, )s' and split to make a tuple
                            value1, value2 = tuple(
                                map(float, value)
                            )  # Convert each element to float
                            reversed_attrs["--noise-range-min"] = value1
                            reversed_attrs["--noise-range-max"] = value2
                            continue
                        else:
                            value = value[:-1]  # Remove the 's'

                    reversed_attrs[original_key] = (
                        value  # Use the original key with the unchanged value
                    )

        reversed_attrs["--image-output-folder"] = reversed_attrs[
            "--image-output-folder"
        ][:-1]

        return Config.build_config_string_from_dict(reversed_attrs), reversed_attrs

    @classmethod
    def build_config_string_from_dict(cls, config_dict):
        # Join the dictionary into a string with each key-value pair separated by spaces
        return " ".join(f"{key}={value}" for key, value in config_dict.items())

    @classmethod
    def print_configurations(cls):
        print("Current Configuration:")
        for attr, value in cls.__dict__.items():
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                print(f"{attr} = {value}")

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
    def set_skips(cls, skips):
        cls.SKIPS = skips

    @classmethod
    def set_encoder(cls, encode):
        cls.ENCODER = encode

    @classmethod
    def set_flip_colour(cls, colour):
        cls.FLIP_COLOUR = colour

    @classmethod
    def set_flip_horizontal(cls, horizontal):
        cls.FLIP_HORIZONTAL = horizontal

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
    parser.add_argument("--skips", type=bool, help="Enable skips")
    parser.add_argument(
        "--flip-colour", type=bool, help="Enable for colour channel flipping"
    )
    parser.add_argument(
        "--flip-horizontal", type=bool, help="Enable for horizontal flipping"
    )
    parser.add_argument(
        "--encoder", type=bool, help="Enable for encoder, false for diffusion"
    )
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
    if args.skips is not None:
        Config.set_skips(args.skips)
    if args.encoder is not None:
        Config.set_encoder(args.encoder)
    if args.flip_colour is not None:
        Config.set_flip_colour(args.flip_colour)
    if args.flip_horizontal is not None:
        Config.set_flip_horizontal(args.flip_horizontal)
    if args.learning_rate is not None:
        Config.set_learning_rate(args.learning_rate)
    if args.image_output_folder is not None:
        Config.set_image_output_folder(args.image_output_folder)

    Config.print_configurations()
