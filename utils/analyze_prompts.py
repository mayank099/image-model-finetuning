"""
Utility functions for analyzing and generating prompts.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from utils.pipeline_utils import load_env_file
from model_finetuning import ConfigManager, PromptAnalyzer


def analyze_prompts(
    descriptions_dir: Optional[str] = None,
    subject_name: Optional[str] = None,
    config_path: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Analyze existing descriptions to extract patterns.

    Args:
        descriptions_dir: Directory containing descriptions
        subject_name: Name of the subject in the images
        config_path: Path to config file

    Returns:
        Tuple of (analysis dictionary, list of descriptions)
    """
    # Load environment variables
    load_env_file()

    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if descriptions_dir:
        config.set("processed_images_dir", descriptions_dir)
    if subject_name:
        config.set("subject_name", subject_name)

    # Initialize prompt analyzer
    analyzer = PromptAnalyzer(config)

    # Analyze descriptions
    descriptions_dir = config.get("processed_images_dir")
    print(f"Analyzing descriptions in {descriptions_dir}")
    analysis, descriptions = analyzer.analyze_descriptions(descriptions_dir)

    # Print analysis results
    if analysis:
        print("\nAnalysis Results:")
        print(f"Total descriptions: {analysis['total_descriptions']}")
        print(f"Common photo types: {analysis['common_photo_types']}")
        print(f"Common settings: {analysis['common_settings']}")
        print(f"Common poses: {analysis['common_poses']}")
        print(f"Common attire: {analysis['common_attire']}")
    else:
        print("No descriptions found to analyze.")

    return analysis, descriptions


def generate_new_prompts(
    analysis: Optional[Dict[str, Any]] = None,
    descriptions: Optional[List[str]] = None,
    descriptions_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    num_prompts: int = 5,
    subject_name: Optional[str] = None,
    openai_model: str = "gpt-4o-2024-08-06",
    config_path: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    Generate new prompts based on analysis of existing descriptions.

    Args:
        analysis: Analysis dictionary (from analyze_prompts)
        descriptions: List of descriptions (from analyze_prompts)
        descriptions_dir: Directory containing descriptions
        output_file: File to save generated prompts
        num_prompts: Number of prompts to generate
        subject_name: Name of the subject in the images
        openai_model: OpenAI model to use
        config_path: Path to config file

    Returns:
        Tuple of (list of prompts, path to output file)
    """
    # Load environment variables
    load_env_file()

    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        return [], ""

    # Create configuration
    config = ConfigManager(config_path)

    # Override config with function parameters
    if descriptions_dir:
        config.set("processed_images_dir", descriptions_dir)
    if subject_name:
        config.set("subject_name", subject_name)
    if openai_model:
        config.set("openai_model", openai_model)
    if num_prompts:
        config.set("num_prompts", num_prompts)

    # Initialize prompt analyzer
    analyzer = PromptAnalyzer(config)

    # Get analysis if not provided
    if analysis is None or descriptions is None:
        analysis, descriptions = analyze_prompts(
            descriptions_dir=descriptions_dir,
            subject_name=subject_name,
            config_path=config_path
        )

    # Generate new prompts
    if not analysis:
        print("No analysis available. Cannot generate prompts.")
        return [], ""

    print(f"\nGenerating {num_prompts} new prompts...")
    prompts = analyzer.generate_new_prompts(
        analysis=analysis,
        descriptions=descriptions,
        num_prompts=num_prompts
    )

    # Save prompts
    if output_file:
        config.set("output_dir", os.path.dirname(output_file))
    output_file = analyzer.save_prompts(prompts, output_file)

    # Print generated prompts
    print("\nGenerated Prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")

    return prompts, output_file


if __name__ == "__main__":
    # This allows the script to be run directly
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze descriptions and generate prompts")
    parser.add_argument("--dir", help="Directory containing descriptions")
    parser.add_argument("--output", help="File to save generated prompts")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of prompts to generate")
    parser.add_argument("--subject", help="Name of the subject in the images")
    parser.add_argument("--model", help="OpenAI model to use",
                        default="gpt-4o-2024-08-06")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze, don't generate prompts")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Analyze descriptions
    analysis, descriptions = analyze_prompts(
        descriptions_dir=args.dir,
        subject_name=args.subject,
        config_path=args.config
    )

    # Generate prompts if requested
    if not args.analyze_only and analysis:
        generate_new_prompts(
            analysis=analysis,
            descriptions=descriptions,
            output_file=args.output,
            num_prompts=args.count,
            subject_name=args.subject,
            openai_model=args.model,
            config_path=args.config
        )
