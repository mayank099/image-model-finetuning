"""
Utility functions for training models on Replicate.
"""

import os
from typing import Optional, Any, Tuple
from utils.pipeline_utils import load_env_file
from model_finetuning import ConfigManager, ModelTrainer


def create_training_data(
    images_dir: Optional[str] = None,
    output_zip: Optional[str] = None,
    config_path: Optional[str] = None
) -> str:
    """
    Create a ZIP file of training data from processed images and descriptions.

    Args:
        images_dir: Directory containing processed images and descriptions
        output_zip: Path to save the ZIP file
        config_path: Path to config file

    Returns:
        Path to the created ZIP file
    """
    # Load environment variables
    load_env_file()

    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if images_dir:
        config.set("processed_images_dir", images_dir)

    # Initialize model trainer
    trainer = ModelTrainer(config)

    # Create training data ZIP
    images_dir = config.get("processed_images_dir")
    print(f"Creating training data ZIP from {images_dir}")
    training_zip = trainer.create_training_data_zip(
        images_dir=images_dir,
        output_zip=output_zip
    )

    print(f"Training data ZIP created: {training_zip}")
    return training_zip


def train_model(
    training_zip: Optional[str] = None,
    images_dir: Optional[str] = None,
    replicate_username: Optional[str] = None,
    model_name: Optional[str] = None,
    model_visibility: Optional[str] = None,
    model_description: Optional[str] = None,
    training_steps: Optional[int] = None,
    config_path: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Train a model on Replicate using the provided training data.

    Args:
        training_zip: Path to training data ZIP
        images_dir: Directory containing processed images (if training_zip not provided)
        replicate_username: Replicate username
        model_name: Name for the model
        model_visibility: Model visibility ("public" or "private")
        model_description: Description for the model
        training_steps: Number of training steps
        config_path: Path to config file

    Returns:
        Tuple of (model object, training object)
    """
    # Load environment variables
    load_env_file()

    # Check if REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERROR: REPLICATE_API_TOKEN environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        return None, None

    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if replicate_username:
        config.set("replicate_username", replicate_username)
    if model_name:
        config.set("model_name", model_name)
    if model_visibility:
        config.set("model_visibility", model_visibility)
    if model_description:
        config.set("model_description", model_description)
    if training_steps:
        config.set("training_steps", training_steps)
    if images_dir:
        config.set("processed_images_dir", images_dir)

    # Initialize model trainer
    trainer = ModelTrainer(config)

    try:
        # Create the training data ZIP if not provided
        if not training_zip:
            training_zip = create_training_data(
                images_dir=images_dir,
                config_path=config_path
            )

        # Create model on Replicate
        print("\nCreating model on Replicate...")
        model = trainer.create_flux_model()
        print(f"Model created: {model.name}")
        print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")

        # Start training
        print("\nStarting model training...")
        training = trainer.train_flux_model(
            model=model,
            training_zip=training_zip,
            steps=config.get("training_steps")
        )
        print(f"Training started: {training.status}")
        print(f"Training URL: https://replicate.com/p/{training.id}")

        print("\nTraining is now running in the background on Replicate.")
        print("You can check its status using the URL above.")

        return model, training

    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None


if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse

    parser = argparse.ArgumentParser(description="Train a model on Replicate")
    parser.add_argument(
        "--images-dir", help="Directory containing processed images")
    parser.add_argument("--training-zip", help="Path to training data ZIP")
    parser.add_argument("--username", help="Replicate username")
    parser.add_argument("--model-name", help="Name for the model")
    parser.add_argument(
        "--visibility", choices=["public", "private"], default="public", help="Model visibility")
    parser.add_argument("--description", help="Description for the model")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    parser.add_argument("--create-zip-only", action="store_true",
                        help="Only create training data ZIP")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Create training data ZIP if requested
    if args.create_zip_only:
        create_training_data(
            images_dir=args.images_dir,
            output_zip=args.training_zip,
            config_path=args.config
        )
    else:
        # Train model
        train_model(
            training_zip=args.training_zip,
            images_dir=args.images_dir,
            replicate_username=args.username,
            model_name=args.model_name,
            model_visibility=args.visibility,
            model_description=args.description,
            training_steps=args.steps,
            config_path=args.config
        )
