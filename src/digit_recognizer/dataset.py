import os

import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import cv2

from src.utils import add_to_log

from typing import Dict, Any, List


class Dataset(object):
    """Loads the dataset based on model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def load_data(self) -> None:
        """Loads original train CSV file as a dataframe.

        Loads original train CSV file as a dataframe.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        self.original_train_data = pd.read_csv(
            "{}/data/raw_data/digit_recognizer/train.csv".format(
                self.home_directory_path
            )
        )

        add_to_log(
            "No. of examples in the original train data: {}".format(
                len(self.original_train_data)
            )
        )
        add_to_log("")

    def split_dataset(self) -> None:
        """Splits original train data into new train, validation & test data.

        Splits original train data into new train, validation & test data.

        Args:
            None.

        Returns:
            None.
        """
        # Computes number of examples in the train, validation and test datasets.
        self.n_total_examples = len(self.original_train_data)
        self.n_validation_examples = int(
            self.model_configuration["dataset"]["split_percentage"]["validation"]
            * self.n_total_examples
        )
        self.n_test_examples = int(
            self.model_configuration["dataset"]["split_percentage"]["test"]
            * self.n_total_examples
        )
        self.n_train_examples = (
            self.n_total_examples - self.n_validation_examples - self.n_test_examples
        )

        # Shuffles the original train data.
        self.original_train_data = shuffle(self.original_train_data, random_state=2)

        # Splits the original train data into new train, validation & test data.
        self.new_validation_data = self.original_train_data[
            : self.n_validation_examples
        ]
        self.new_test_data = self.original_train_data[
            self.n_validation_examples : self.n_test_examples
            + self.n_validation_examples
        ]
        self.new_train_data = self.original_train_data[
            self.n_test_examples + self.n_validation_examples :
        ]

        add_to_log(
            "No. of examples in the new train data: {}".format(self.n_train_examples)
        )
        add_to_log(
            "No. of examples in the new validation data: {}".format(
                self.n_validation_examples
            )
        )
        add_to_log(
            "No. of examples in the new test data: {}".format(self.n_test_examples)
        )
        add_to_log("")

    def shuffle_slice_dataset(self) -> None:
        """Converts split data into tensor dataset & slices them based on batch size.

        Converts split data into input & target data. Zips the input & target data, and slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips images & classes into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_train_data.drop(columns=["label"]),
                list(self.new_train_data["label"]),
            )
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_validation_data.drop(columns=["label"]),
                list(self.new_validation_data["label"]),
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_test_data.drop(columns=["label"]),
                list(self.new_test_data["label"]),
            )
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = self.n_train_examples // self.batch_size
        self.n_validation_steps_per_epoch = (
            self.n_validation_examples // self.batch_size
        )
        self.n_test_steps_per_epoch = self.n_test_examples // self.batch_size

        add_to_log(
            "No. of train steps per epoch: {}".format(self.n_train_steps_per_epoch)
        )
        add_to_log(
            "No. of validation steps per epoch: {}".format(
                self.n_validation_steps_per_epoch
            )
        )
        add_to_log(
            "No. of test steps per epoch: {}".format(self.n_test_steps_per_epoch)
        )
        add_to_log("")

    def invert_image(self, image: np.ndarray) -> np.ndarray:
        """Inverts the image from black & white to white & black.

        Inverts the image from black & white to white & black.

        Args:
            image: A NumPy array for the input image.

        Returns:
            A NumPy array for inverted version of the input image.
        """
        # Checks type & values of arguments.
        assert isinstance(
            image, np.ndarray
        ), "Variable image should be of type 'np.ndarray'."
        assert len(image.shape) == 3, "Variable image should be 3 dimensional."

        # Inverts the image from black & white to white & black
        _, inverted_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        return inverted_image
