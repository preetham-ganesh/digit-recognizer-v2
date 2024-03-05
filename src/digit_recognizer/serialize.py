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

        @tf.function(
            input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
        )
        def predict(self, inputs: tf.Tensor) -> tf.Tensor:
            """Input image(s) are passed through the model for prediction.

            Input image(s) are passed through the model for prediction.

            Args:
                inputs: A tensor for the processed image(s) based on model requirements.

            Returns:
                A tensor with prediction for each processed image passed as input.
            """
            prediction = self.model([inputs], False, None)
            return prediction

    # Exports trained tensorflow model as tensorflow module for serving.
    exported_model = ExportModel(model)

    # Predicts output for the sample input using the Exported model.
    output_0 = exported_model.predict(
        tf.ones(
            (
                10,
                trainer.model_configuration["model"]["final_image_height"],
                trainer.model_configuration["model"]["final_image_width"],
                trainer.model_configuration["model"]["n_channels"],
            )
        )
    )

    # Saves the tensorflow object created from the loaded model.
    home_directory_path = os.getcwd()
    tf.saved_model.save(
        exported_model,
        "{}/models/digit_recognizer/v{}/serialized".format(
            home_directory_path, model_version
        ),
    )

    # Loads the serialized model to check if the loaded model is callable.
    exported_model = tf.saved_model.load(
        "{}/models/digit_recognizer/v{}/serialized".format(
            home_directory_path, model_version
        ),
    )
    output_1 = exported_model.predict(
        tf.ones(
            (
                10,
                trainer.model_configuration["model"]["final_image_height"],
                trainer.model_configuration["model"]["final_image_width"],
                trainer.model_configuration["model"]["n_channels"],
            )
        )
    )

    # Checks if the shape between output from saved & loaded models matches.
    assert (
        output_0.shape == output_1.shape
    ), "Shape does not match between the output from saved & loaded models."
    add_to_log("Finished serializing model & configuration files.")
    add_to_log("")


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
