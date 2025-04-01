"""
Configuration management for the image model fine-tuning pipeline.
"""

import os
import configparser
from typing import Any, Dict, Optional, Tuple, Union


class ConfigManager:
    """Manages configuration settings for the pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            # Directories
            "raw_images_dir": os.path.expanduser("~/Desktop/Image_Model_Training/image-model-finetuning/data/raw_images"),
            "processed_images_dir": os.path.expanduser("~/Desktop/Image_Model_Training/image-model-finetuning/data/processed_images"),
            "output_dir": os.path.expanduser("~/Desktop/Image_Model_Training/image-model-finetuning/data/output"),

            # Image processing
            "output_size": (1024, 1024),
            "padding_color": (0, 0, 0),  # Black

            # Model training
            "replicate_username": "mayank099",
            "model_name": "flux-new-ft",  # Update Model Name (Pass in Params)
            "model_visibility": "public",
            "model_description": "FLUX.1 finetuned on personal photos",
            "training_steps": 1000,

            # Prompt generation
            "subject_name": "Mayank",
            "num_prompts": 5,
            "openai_model": "gpt-4o-2024-08-06"
        }

        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file.

        Args:
            config_path: Path to configuration file
        """
        config = configparser.ConfigParser()
        config.read(config_path)

        # Process each section
        for section in config.sections():
            for key, value in config[section].items():
                full_key = f"{section}_{key}" if section != "DEFAULT" else key
                self.config[full_key] = self._parse_value(value)

    def _parse_value(self, value: str) -> Any:
        """
        Parse string values into appropriate types.

        Args:
            value: String value to parse

        Returns:
            Parsed value in appropriate type
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Try to convert to tuple (for sizes and colors)
        if value.startswith('(') and value.endswith(')'):
            try:
                return eval(value)  # Use with caution, only for parsing tuples
            except:
                pass

        # Return as string
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key doesn't exist

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def ensure_directories(self) -> Dict[str, Any]:
        """
        Ensure all necessary directories exist.

        Returns:
            Dictionary of configuration values
        """
        for dir_key in ["raw_images_dir", "processed_images_dir", "output_dir"]:
            os.makedirs(self.config[dir_key], exist_ok=True)

        return self.config

    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to a file.

        Args:
            config_path: Path to save configuration
        """
        config = configparser.ConfigParser()
        config["DEFAULT"] = {}

        # Group keys by section
        sections = {}
        for key, value in self.config.items():
            if "_" in key:
                section, option = key.split("_", 1)
                if section not in sections:
                    sections[section] = {}
                sections[section][option] = str(value)
            else:
                config["DEFAULT"][key] = str(value)

        # Add sections
        for section, options in sections.items():
            config[section] = options

        # Write to file
        with open(config_path, 'w') as configfile:
            config.write(configfile)
