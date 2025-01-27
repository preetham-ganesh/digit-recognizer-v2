import os
import time

import tensorflow as tf
import mlflow

from src.utils import (
    load_json_file,
    check_directory_path_existence,
)
from src.dataset import Dataset
from src.model import Model


class Train(object):
    """Trains the digit recognition model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for current version.

        Loads the model configuration file for current version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs"
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads dataset based on model configuration.

        Loads dataset based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads original train CSV file as a dataframe.
        self.dataset.load_data()

        # Splits original train data into new train, validation & test data.
        self.dataset.split_dataset()

        # Converts split data tensorflow dataset and slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

    def load_model(self) -> None:
        """Loads model & other utilies for training.

        Loads model & other utilies for training.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = Model(self.model_configuration)

        # Builds plottable graph for the model.
        self.model = self.model.build_graph()

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["optimizer"][
                "learning_rate"
            ]
        )

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path, "models", f"v{self.model_version}", "checkpoints"
        )
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=1
        )
        print("Finished loading model for current configuration.")
        print()

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Compiles the model to log the model summary.
        model_summary = list()
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        mlflow.log_text(
            model_summary,
            os.path.join(f"v{self.model_version}", "model_summary.txt"),
        )

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            os.path.join("models", f"v{self.model_version}", "reports")
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                self.model,
                os.path.join(self.reports_directory_path, "model_plot.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                os.path.join(self.reports_directory_path, "model_plot.png"),
                f"v{self.model_version}",
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for the the actual values for the current batch.
            predicted_batch: A tensor for the predicted values for the current batch.

        Returns:
            A tensor for the loss for the current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for the current batch using actual values and predicted values.
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = self.loss_object(target_batch, predicted_batch)
        return loss

    def compute_accuracy(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual & predicted values.

        Computes accuracy for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor which contains the actual values for the current batch.
            predicted_batch: A tensor which contains the predicted values for the current batch.

        Returns:
            A tensor for the accuracy of current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes accuracy for the current batch using actual values and predicted values.
        accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
        return accuracy

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains model using current input & target batches.

        Trains model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for training the model.
            target_batch: A tensor for the target text from the current batch for training and validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        with tf.GradientTape() as tape:
            predictions = self.model([input_batch], training=True, masks=None)
            loss = self.compute_loss(target_batch, predictions[0])
            accuracy = self.compute_accuracy(target_batch, predictions[0])

        # Computes gradients using loss and model variables.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Uses optimizer to apply the computed gradients on the combined model variables.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes batch metrics and appends it to main metrics.
        self.train_loss(loss)
        self.train_accuracy(accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates model using current input & target batches.

        Validates model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for validating the model.
            target_batch: A tensor for the target text from the current batch for validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        predictions = self.model([input_batch], training=False, masks=None)
        loss = self.compute_loss(target_batch, predictions[0])
        accuracy = self.compute_accuracy(target_batch, predictions[0])

        # Computes batch metrics and appends it to main metrics.
        self.validation_loss(loss)
        self.validation_accuracy(accuracy)

    def reset_metrics_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()
        self.train_accuracy.reset_state()
        self.validation_accuracy.reset_state()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (images, labels) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                images.numpy(), labels.numpy()
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()

            print(
                "Epoch={}, Batch={}, Train loss={}, Train accuracy={}, Time taken={} sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "train_loss": self.train_loss.result().numpy(),
                "train_accuracy": self.train_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using validation dataset for current epoch.

        Validates the model using validation dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (images, labels) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                images.numpy(), labels.numpy()
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()

            print(
                "Epoch={}, Batch={}, Validation loss={}, Validation accuracy={}, Time taken={} sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.validation_accuracy.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "validation_loss": self.validation_loss.result().numpy(),
                "validation_accuracy": self.validation_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current epoch.

        Saves the model after checking performance metrics in current epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print("Checkpoint saved at {}.".format(self.checkpoint_directory_path))

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous epoch.

        Stops the model from learning further if the performance has not improved from previous epoch.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > str(
            round(self.validation_loss.result().numpy(), 3)
        ):
            self.patience_count = 0
            print(
                "Best validation loss changed from {} to {}".format(
                    str(self.best_validation_loss),
                    str(round(self.validation_loss.result().numpy(), 3)),
                )
            )
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < 2:
            self.patience_count += 1
            print("Best validation loss did not improve.")
            print("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes TensorFlow trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for epoch in range(self.model_configuration["model"]["epochs"]):
            epoch_start_time = time.time()

            # Resets states for training and validation metrics before the start of each epoch.
            self.reset_metrics_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)

            epoch_end_time = time.time()
            print(
                "Epoch={}, Train loss={}, Validation loss={}, Train Accuracy={}, Validation Accuracy={}, "
                "Time taken={} sec.".format(
                    epoch + 1,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    str(round(self.validation_accuracy.result().numpy(), 3)),
                    round(epoch_end_time - epoch_start_time, 3),
                )
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                print(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                print()
                break
            print()

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_metrics_trackers()

        # Restore latest saved checkpoint if available.
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_directory_path)
        )

        # Iterates across batches in the train dataset.
        for batch, (images, labels) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                images.numpy(), labels.numpy()
            )

            # Tests the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)

        print(
            "Test loss={}.".format(str(round(self.validation_loss.result().numpy(), 3)))
        )
        print(
            "Test accuracy={}.".format(
                str(round(self.validation_accuracy.result().numpy(), 3))
            ),
        )
        print()

        # Logs test metrics for current epoch.
        mlflow.log_metrics(
            {
                "test_loss": self.validation_loss.result().numpy(),
                "test_accuracy": self.validation_accuracy.result().numpy(),
            }
        )

    def serialize_model(self) -> None:
        """Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Args:
            None.

        Returns:
            None.
        """
        # Defines input shape for exported model's input signature.
        input_shape = [
            None,
            self.model_configuration["model"]["final_image_height"],
            self.model_configuration["model"]["final_image_width"],
            self.model_configuration["model"]["n_channels"],
        ]

        class ExportModel(tf.Module):
            """Exports trained tensorflow model as tensorflow module for serving."""

            def __init__(self, model: tf.keras.Model) -> None:
                """Initializes the variables in the class.

                Initializes the variables in the class.

                Args:
                    model: A tensorflow model for the model trained with latest checkpoints.

                Returns:
                    None.
                """
                # Asserts type of input arguments.
                assert isinstance(
                    model, tf.keras.Model
                ), "Variable model should be of type 'tensorflow.keras.Model'."

                # Initializes class variables.
                self.model = model

            @tf.function(
                input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
            )
            def predict(self, images: tf.Tensor) -> tf.Tensor:
                """Input image is passed through the model for prediction.

                Input image is passed through the model for prediction.

                Args:
                    images: A tensor for the processed image for which the model should predict the result.

                Return:
                    An integer for the number predicted by the model for the current image.
                """
                prediction = self.model([images], training=False, masks=None)
                return prediction

        # Exports trained tensorflow model as tensorflow module for serving.
        exported_model = ExportModel(self.model)

        # Predicts output for the sample input using the Exported model.
        output_0 = exported_model.predict(
            tf.ones(
                (
                    10,
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        )

        # Saves the tensorflow object created from the loaded model.
        home_directory_path = os.getcwd()
        tf.saved_model.save(
            exported_model,
            "{}/models/v{}/serialized".format(home_directory_path, self.model_version),
        )

        # Loads the serialized model to check if the loaded model is callable.
        exported_model = tf.saved_model.load(
            "{}/models/v{}/serialized".format(home_directory_path, self.model_version),
        )
        output_1 = exported_model.predict(
            tf.ones(
                (
                    10,
                    self.model_configuration["model"]["final_image_height"],
                    self.model_configuration["model"]["final_image_width"],
                    self.model_configuration["model"]["n_channels"],
                )
            )
        )

        # Checks if the shape between output from saved & loaded models matches.
        assert (
            output_0[0].shape == output_1[0].shape
        ), "Shape does not match between the output from saved & loaded models."
        print("Finished serializing model & configuration files.")
        print()

        # Logs serialized model as artifact.
        mlflow.log_artifacts(
            "{}/models/v{}/serialized".format(home_directory_path, self.model_version),
            "v{}/model".format(self.model_configuration["version"]),
        )

        # Logs updated model configuration as artifact.
        mlflow.log_dict(
            self.model_configuration,
            "v{}/model_configuration.json".format(self.model_version),
        )
