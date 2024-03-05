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


import tensorflow as tf

from src.utils import create_log, add_to_log, set_physical_devices_memory_limit
from src.digit_recognizer.train import Train


def serialize_model(model_version: str) -> None:
    """Serializes model files in the serialized model directory.

    Serializes model files in the serialized model directory.

    Args:
        model_version: A string for the version of the model about to be serialized.

    Returns:
        None.
    """
    # Checks type & values of arguments.
    assert isinstance(
        model_version, str
    ), "Variable model_version should be of type 'str'."

    # Creates an object for the Train class.
    trainer = Train(model_version)

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads the model with latest checkpoint.
    trainer.load_model("predict")

    # Generates summary & plot for loaded model.
    trainer.generate_model_summary_and_plot(False)

    # Builds plottable graph for the model.
    model = trainer.model.build_graph()

    # Defines input shape for exported model's input signature.
    input_shape = [
        None,
        trainer.model_configuration["model"]["final_image_height"],
        trainer.model_configuration["model"]["final_image_width"],
        trainer.model_configuration["model"]["n_channels"],
    ]

    class ExportModel(tf.Module):
        """Exports a trained TensorFlow model as a TensorFlow module for serving."""

        def __init__(self, model: tf.keras.Model) -> None:
            """Initializes variables in the ExportModel class.

            Initializes variables in the ExportModel class.

            Args:
                model: A tensorflow model with the latest trained checkpoint.

            Returns:
                None.
            """
            # Checks type & values of arguments.
            assert isinstance(
                model, tf.keras.Model
            ), "Variable model should be of type 'tensorflow.keras.model'."

            # Initializes class variables.
            self.model = model


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
