"""
Command-line interface for the image model fine-tuning pipeline.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

from .pipeline import PipelineManager
from utils.pipeline_utils import load_env_file


def main():
    """Entry point for the command-line interface."""
    # Load environment variables from .env file
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Image Model Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        choices=["process", "describe", "analyze",
                 "generate", "train", "full"],
        default="full",
        help="Pipeline mode to run"
    )

    parser.add_argument(
        "--input",
        help="Input directory path for raw images"
    )

    parser.add_argument(
        "--output",
        help="Output directory path for results"
    )

    parser.add_argument(
        "--steps",
        type=int,
        help="Number of training steps"
    )

    parser.add_argument(
        "--copy-text",
        action="store_true",
        help="Copy text files associated with images (if they exist)"
    )

    parser.add_argument(
        "--template",
        help="Path to a template file for generating descriptions"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing descriptions"
    )

    parser.add_argument(
        "--subject",
        help="Name of the subject in the images"
    )

    parser.add_argument(
        "--prompts",
        type=int,
        help="Number of prompts to generate"
    )

    parser.add_argument(
        "--username",
        help="Replicate username for model training"
    )

    parser.add_argument(
        "--model-name",
        help="Name for the trained model"
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = PipelineManager(args.config)

        # Override config with command-line arguments
        if args.input:
            pipeline.config_manager.set("raw_images_dir", args.input)
        if args.output:
            pipeline.config_manager.set("output_dir", args.output)
        if args.steps:
            pipeline.config_manager.set("training_steps", args.steps)
        if args.subject:
            pipeline.config_manager.set("subject_name", args.subject)
        if args.prompts:
            pipeline.config_manager.set("num_prompts", args.prompts)
        if args.username:
            pipeline.config_manager.set("replicate_username", args.username)
        if args.model_name:
            pipeline.config_manager.set("model_name", args.model_name)

        # Set config options from arguments
        if args.overwrite:
            pipeline.config_manager.set("overwrite_descriptions", True)

        # Run the specified pipeline mode
        if args.mode == "process":
            pipeline.run_image_processing(copy_text_files=args.copy_text)
        elif args.mode == "describe":
            pipeline.generate_descriptions(template_file=args.template)
        elif args.mode == "analyze":
            pipeline.run_prompt_analysis()
        elif args.mode == "generate":
            analysis, descriptions = pipeline.run_prompt_analysis()
            pipeline.run_prompt_generation(analysis, descriptions)
        elif args.mode == "train":
            pipeline.run_model_training()
        elif args.mode == "full":
            # Pass the text file copy preference to the full pipeline
            copy_text = args.copy_text if hasattr(args, 'copy_text') else False
            pipeline.config_manager.set("copy_text_files", copy_text)
            pipeline.run_full_pipeline(template_file=args.template)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
