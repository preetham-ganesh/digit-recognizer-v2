import os

import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import cv2
import random

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

        # Converts 2D image into 3D by stacking it.
        new_image = np.zeros((image.shape[0], image.shape[1], 3))
        new_image[:, :, 0] = image
        new_image[:, :, 1] = image
        new_image[:, :, 2] = image

        # Inverts the image from black & white to white & black.
        _, inverted_image = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY_INV)

        # Extracts the 1st channel from the inverted image.
        inverted_2d_image = inverted_image[:, :, 0]
        return inverted_2d_image

    def load_input_target_batches(
        self, images: np.ndarray, labels: np.ndarray
    ) -> List[tf.Tensor]:
        """Load input & target batchs for images & labels.

        Load input & target batchs for images & labels.

        Args:
            images: A NumPy array for images in current batch.
            labels: A NumPy array for labels in current batch.

        Returns:
            A list of tensors for the input & target batches generated from images & labels.
        """
        # Checks types & values of arguments.
        assert isinstance(
            images, np.ndarray
        ), "Variable images should be of type 'np.ndarray'."
        assert isinstance(
            labels, np.ndarray
        ), "Variable labels should be of type 'np.ndarray'."

        # Reshapes input batch into (batch_size, final_image_height, final_image_width).
        images = images.reshape(
            (
                self.model_configuration["model"]["batch_size"],
                self.model_configuration["model"]["final_image_height"],
                self.model_configuration["model"]["final_image_width"],
            )
        )

        # Iterates across images in the batch.
        n_images = len(images)
        for image_id in range(n_images):

            # Checks if probability is greater than 0.5. If greater then inverts black & white image -> white & black.
            if random.random() >= 0.5:
                images[image_id, :, :] = self.invert_image(images[image_id])

        # Adds extra dimension to the images array.
        images = np.expand_dims(images, axis=3)

        # Converts images into tensor, and converts pixels into 0 - 1 range.
        input_batch = tf.convert_to_tensor(images)
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        input_batch /= 255.0

        # Converts labels into categorical tensor.
        target_batch = tf.keras.utils.to_categorical(
            labels, num_classes=self.model_configuration["model"]["n_classes"]
        )
        target_batch = tf.convert_to_tensor(target_batch, dtype=tf.int8)
        return [input_batch, target_batch]
