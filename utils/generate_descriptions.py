"""
Utility functions for generating descriptions for images.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from model_finetuning import ConfigManager, PromptAnalyzer


def generate_descriptions(
    image_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    template_file: Optional[str] = None,
    subject_name: Optional[str] = None,
    openai_model: str = "gpt-4-vision-preview",
    overwrite: bool = False,
    config_path: Optional[str] = None
) -> List[str]:
    """
    Generate descriptions for all images in the specified directory.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save descriptions (defaults to image_dir)
        template_file: Path to description template
        subject_name: Name of the subject in the images
        openai_model: OpenAI model to use
        overwrite: Whether to overwrite existing descriptions
        config_path: Path to config file

    Returns:
        List of generated descriptions
    """
    # Load environment variables
    load_dotenv()

    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        return []

    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if image_dir:
        config.set("processed_images_dir", image_dir)
    if output_dir:
        config.set("output_dir", output_dir)
    if subject_name:
        config.set("subject_name", subject_name)
    if openai_model:
        config.set("openai_model", openai_model)

    config.set("overwrite_descriptions", overwrite)

    # Find default template if none provided
    if not template_file:
        package_dir = os.path.dirname(os.path.dirname(__file__))
        default_template = os.path.join(
            package_dir, "templates", "description_template.txt")
        if os.path.exists(default_template):
            template_file = default_template

    # Initialize prompt analyzer
    analyzer = PromptAnalyzer(config)

    # Generate descriptions
    image_dir = config.get("processed_images_dir")
    output_dir = config.get("output_dir") or image_dir

    print(f"Generating descriptions for images in {image_dir}")
    descriptions = analyzer.generate_descriptions(
        image_dir=image_dir,
        template_file=template_file,
        output_dir=output_dir
    )

    # Print sample descriptions
    if descriptions:
        print("\nGenerated descriptions (sample):")
        for i, desc in enumerate(descriptions[:3], 1):
            print(f"{i}. {desc}")

        if len(descriptions) > 3:
            print(f"... and {len(descriptions) - 3} more")

    return descriptions


if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate descriptions for images")
    parser.add_argument("--image-dir", help="Directory containing images")
    parser.add_argument("--output-dir", help="Directory to save descriptions")
    parser.add_argument("--template", help="Path to description template")
    parser.add_argument("--subject", help="Name of the subject in the images")
    parser.add_argument("--model", help="OpenAI model to use",
                        default="gpt-4-vision-preview")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing descriptions")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Generate descriptions
    generate_descriptions(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        template_file=args.template,
        subject_name=args.subject,
        openai_model=args.model,
        overwrite=args.overwrite,
        config_path=args.config
    )
