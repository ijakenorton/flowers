import sys
from config import Config

# Pipe file name into stdin, will output the file args needed to recreate the
# result
if __name__ == "__main__":
    input_data = sys.stdin.read().strip()
    args, arg_dict = Config.parse_config_filename(input_data)

    print(args)
