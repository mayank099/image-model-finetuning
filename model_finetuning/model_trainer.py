"""
Model training functionality for the fine-tuning pipeline.
"""

import os
import zipfile
from typing import Optional, Tuple, Any
import replicate

from .config import ConfigManager


class ModelTrainer:
    """Handles the training of custom image models on Replicate."""

    def __init__(self, config: ConfigManager):
        """
        Initialize the model trainer.

        Args:
            config: Configuration manager instance
        """
        self.config = config

    def create_training_data_zip(self, images_dir: Optional[str] = None,
                                 output_zip: Optional[str] = None) -> str:
        """
        Create a ZIP file from the directory with images and text files.

        Args:
            images_dir: Directory containing processed images (optional)
            output_zip: Path to save ZIP file (optional)

        Returns:
            Path to the created ZIP file
        """
        images_dir = images_dir or self.config.get("processed_images_dir")

        if output_zip is None:
            output_dir = self.config.get("output_dir")
            os.makedirs(output_dir, exist_ok=True)
            output_zip = os.path.join(output_dir, "data.zip")

        # Get absolute path
        output_zip = os.path.abspath(output_zip)

        # Create ZIP file
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, _, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add to ZIP with relative path
                    arcname = os.path.relpath(
                        file_path, os.path.dirname(images_dir))
                    zipf.write(file_path, arcname)

        print(f"Created ZIP file: {output_zip}")
        return output_zip

    def create_flux_model(self) -> Any:
        """
        Create a new model on Replicate for fine-tuning.

        Returns:
            Replicate model object

        Raises:
            ValueError: If configuration is incomplete
        """
        # Get configuration values
        username = self.config.get("replicate_username")
        model_name = self.config.get("model_name")
        visibility = self.config.get("model_visibility")
        description = self.config.get("model_description")

        if not username:
            raise ValueError("Replicate username not configured")

        # Check for API token
        replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not replicate_api_token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set")

        # Create model in Replicate
        model = replicate.models.create(
            owner=username,
            name=model_name,
            visibility=visibility,
            hardware="gpu-t4",  # Replicate will override this for fine-tuned models
            description=description
        )

        print(f"Model created: {model.name}")
        print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")

        return model

    def train_flux_model(self, model: Optional[Any] = None,
                         training_zip: Optional[str] = None,
                         steps: Optional[int] = None) -> Any:
        """
        Start the training process for the Flux model.

        Args:
            model: Replicate model object (optional)
            training_zip: Path to training data ZIP (optional)
            steps: Number of training steps (optional)

        Returns:
            Replicate training object
        """
        if model is None:
            model = self.create_flux_model()

        if training_zip is None:
            training_zip = self.create_training_data_zip()

        steps = steps or self.config.get("training_steps")

        # Start training
        with open(training_zip, "rb") as f:
            training = replicate.trainings.create(
                version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
                input={
                    "input_images": f,
                    "steps": steps,
                },
                destination=f"{model.owner}/{model.name}"
            )

        print(f"Training started: {training.status}")
        print(f"Training URL: https://replicate.com/p/{training.id}")

        return training

    def get_training_status(self, training_id: str) -> str:
        """
        Get the status of a training job.

        Args:
            training_id: ID of the training job

        Returns:
            Status string
        """
        try:
            training = replicate.trainings.get(training_id)
            return training.status
        except Exception as e:
            return f"Error getting training status: {e}"

    def cancel_training(self, training_id: str) -> bool:
        """
        Cancel a training job.

        Args:
            training_id: ID of the training job

        Returns:
            True if successful, False otherwise
        """
        try:
            replicate.trainings.cancel(training_id)
            print(f"Training {training_id} cancelled")
            return True
        except Exception as e:
            print(f"Error cancelling training: {e}")
            return False
