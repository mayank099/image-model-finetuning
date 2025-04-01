"""
Pipeline orchestration for image model fine-tuning.
"""

from typing import Dict, List, Tuple, Any, Optional
import os

from .config import ConfigManager
from .image_processor import ImageProcessor
from .prompt_analyzer import PromptAnalyzer
from .model_trainer import ModelTrainer


class PipelineManager:
    """Manages the end-to-end pipeline for image processing and model training."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.ensure_directories()
        self.image_processor = ImageProcessor(self.config_manager)
        self.prompt_analyzer = PromptAnalyzer(self.config_manager)
        self.model_trainer = ModelTrainer(self.config_manager)

    def run_image_processing(self, input_dir: Optional[str] = None,
                             output_dir: Optional[str] = None,
                             copy_text_files: bool = True) -> str:
        """
        Run the image processing pipeline.

        Args:
            input_dir: Directory containing raw images (optional)
            output_dir: Directory to save processed images (optional)
            copy_text_files: Whether to copy corresponding text files if they exist

        Returns:
            Path to the output directory
        """
        print("\n=== Starting Image Processing ===")
        return self.image_processor.process_images(input_dir, output_dir, copy_text_files)

    def generate_descriptions(self, image_dir: Optional[str] = None,
                              template_file: Optional[str] = None,
                              output_dir: Optional[str] = None) -> List[str]:
        """
        Generate descriptions for all images in the specified directory.

        Args:
            image_dir: Directory containing processed images
            template_file: Path to a text file containing the prompt template
            output_dir: Directory to save the descriptions (defaults to image_dir)

        Returns:
            List of generated descriptions
        """
        print("\n=== Starting Description Generation ===")
        return self.prompt_analyzer.generate_descriptions(image_dir, template_file, output_dir)

    def run_prompt_analysis(self, folder_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Run the prompt analysis pipeline.

        Args:
            folder_path: Directory containing text files (optional)

        Returns:
            Tuple of (analysis dictionary, list of descriptions)
        """
        print("\n=== Starting Prompt Analysis ===")
        analysis, descriptions = self.prompt_analyzer.analyze_descriptions(
            folder_path)

        if analysis:
            print("\nAnalysis of existing descriptions:")
            print(f"Total descriptions: {analysis['total_descriptions']}")
            print(f"Common photo types: {analysis['common_photo_types']}")
            print(f"Common settings: {analysis['common_settings']}")
            print(f"Common poses: {analysis['common_poses']}")
            print(f"Common attire: {analysis['common_attire']}")

        return analysis, descriptions

    def run_prompt_generation(self, analysis: Optional[Dict[str, Any]] = None,
                              descriptions: Optional[List[str]] = None,
                              num_prompts: Optional[int] = None) -> Tuple[List[str], str]:
        """
        Run the prompt generation pipeline.

        Args:
            analysis: Analysis dictionary (optional)
            descriptions: List of descriptions (optional)
            num_prompts: Number of prompts to generate (optional)

        Returns:
            Tuple of (list of prompts, path to output file)
        """
        print("\n=== Starting Prompt Generation ===")

        if analysis is None:
            analysis, descriptions = self.run_prompt_analysis()

        new_prompts = self.prompt_analyzer.generate_new_prompts(
            analysis, descriptions, num_prompts
        )

        print("\nGenerated new prompts:")
        for i, prompt in enumerate(new_prompts, 1):
            print(f"Prompt {i}: {prompt}")

        output_file = self.prompt_analyzer.save_prompts(new_prompts)

        return new_prompts, output_file

    def run_model_training(self, processed_dir: Optional[str] = None,
                           steps: Optional[int] = None) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Run the model training pipeline.

        Args:
            processed_dir: Directory containing processed images (optional)
            steps: Number of training steps (optional)

        Returns:
            Tuple of (model object, training object)
        """
        print("\n=== Starting Model Training ===")

        try:
            # Create the model
            model = self.model_trainer.create_flux_model()

            # Create training data ZIP
            training_zip = self.model_trainer.create_training_data_zip(
                processed_dir)

            # Start training
            training = self.model_trainer.train_flux_model(
                model, training_zip, steps)

            return model, training

        except Exception as e:
            print(f"Error during model training: {e}")
            return None, None

    def run_full_pipeline(self, raw_dir: Optional[str] = None,
                          output_dir: Optional[str] = None,
                          steps: Optional[int] = None,
                          template_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from image processing to model training.

        Args:
            raw_dir: Directory containing raw images (optional)
            output_dir: Directory for final outputs (optional)
            steps: Number of training steps (optional)
            template_file: Path to a text file containing the prompt template (optional)

        Returns:
            Dictionary with results from each stage
        """
        # Update config if parameters provided
        if raw_dir:
            self.config_manager.set("raw_images_dir", raw_dir)
        if output_dir:
            self.config_manager.set("output_dir", output_dir)
        if steps:
            self.config_manager.set("training_steps", steps)

        # Get text file copy preference (default to False since we're generating descriptions)
        copy_text_files = self.config_manager.get("copy_text_files", False)

        # Process images
        processed_dir = self.run_image_processing(
            copy_text_files=copy_text_files)

        # Generate descriptions for the processed images
        descriptions = self.generate_descriptions(processed_dir, template_file)

        # Analyze descriptions and generate additional prompts
        if descriptions:
            analysis, _ = self.run_prompt_analysis(processed_dir)
            prompts, prompt_file = self.run_prompt_generation(
                analysis, descriptions)
        else:
            prompts, prompt_file = [], None

        # Train model
        model, training = self.run_model_training(processed_dir)

        return {
            "processed_dir": processed_dir,
            "descriptions": descriptions,
            "prompts": prompts,
            "prompt_file": prompt_file,
            "model": model,
            "training": training
        }
