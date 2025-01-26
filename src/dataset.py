import os

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random

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
            os.path.join(self.home_directory_path, "data", "raw_data", "train.csv")
        )
        print(
            f"No. of examples in the original train data: {len(self.original_train_data)}"
        )
        print()

    def split_dataset(self) -> None:
        """Splits original train data into new train, validation & test data.

        Splits original train data into new train, validation & test data.

        Args:
            None.

        Returns:
            None.
        """
        # Shuffles the original train data.
        self.original_train_data = shuffle(self.original_train_data, random_state=42)

        # Splits the original train data into new train, validation & test data (in stratified manner).
        train_data, self.new_test_data = train_test_split(
            self.original_train_data,
            test_size=self.model_configuration["dataset"]["split_percentage"]["test"],
            stratify=self.original_train_data["label"],
            random_state=42,
        )
        self.new_train_data, self.new_validation_data = train_test_split(
            train_data,
            test_size=self.model_configuration["dataset"]["split_percentage"][
                "validation"
            ],
            stratify=train_data["label"],
            random_state=42,
        )
        del train_data

        # Stores size of new train, validation and test data.
        self.n_train_examples = len(self.new_train_data)
        self.n_validation_examples = len(self.new_validation_data)
        self.n_test_examples = len(self.new_test_data)

        print(f"No. of examples in the new train data: {self.n_train_examples}")
        print(
            f"No. of examples in the new validation data: {self.n_validation_examples}"
        )
        print(f"No. of examples in the new test data: {self.n_test_examples}")
        print()

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

        print(f"No. of train steps per epoch: {self.n_train_steps_per_epoch}")
        print(f"No. of validation steps per epoch: {self.n_validation_steps_per_epoch}")
        print(f"No. of test steps per epoch: {self.n_test_steps_per_epoch}")
        print()

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
                images[image_id, :, :] = np.where(
                    images[image_id, :, :]
                    > self.model_configuration["model"]["threshold"],
                    0,
                    255,
                )

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
