import os
import sys
import warnings
import argparse
import logging
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


from src.utils import create_log, add_to_log, set_physical_devices_memory_limit


def main():
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version by which the serialized model files should be saved as.",
    )
    args = parser.parse_args()

    # Creates an logger object for storing terminal output.
    create_log("serialize_v{}".format(args.model_version), "logs/digit_recognizer")
    add_to_log("")

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    start_time = time.time()

    # Serializes model files in the serialized model directory.
    serialize_model(args.model_version)

    add_to_log(
        "Finished saving the serialized model & its files in {} sec.".format(
            round(time.time() - start_time, 3)
        )
    )
    add_to_log("")


if __name__ == "__main__":
    main()
