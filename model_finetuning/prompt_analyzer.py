"""
Generate image descriptions using OpenAI.
"""

import os
import glob
import re
import random
from PIL import Image
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from openai import OpenAI

from .config import ConfigManager


class PromptAnalyzer:
    """Generates image descriptions using AI and manages prompt creation."""

    def __init__(self, config: ConfigManager):
        """
        Initialize the prompt analyzer.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.openai_client = self._initialize_openai()

    def _initialize_openai(self) -> Optional[OpenAI]:
        """
        Initialize the OpenAI client.

        Returns:
            OpenAI client instance or None if API key is not set
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("Description generation features will not work without an API key.")
            return None

        return OpenAI(api_key=api_key)

    def generate_descriptions(self,
                              image_dir: Optional[str] = None,
                              template_file: Optional[str] = None,
                              output_dir: Optional[str] = None) -> List[str]:
        """
        Generate descriptions for all images in the specified directory.

        Args:
            image_dir: Directory containing processed images
            template_file: Path to a text file containing the prompt template (optional)
            output_dir: Directory to save the descriptions (defaults to image_dir)

        Returns:
            List of generated descriptions
        """
        if not self.openai_client:
            print("OpenAI client not initialized. Cannot generate descriptions.")
            return []

        # Use default directories if not specified
        image_dir = image_dir or self.config.get("processed_images_dir")
        output_dir = output_dir or image_dir

        # Get all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        if not image_paths:
            print(f"No images found in {image_dir}")
            return []

        print(f"Found {len(image_paths)} images for description generation")

        # Get the prompt template
        prompt_template = self._get_prompt_template(template_file)

        # Get subject name from config
        subject_name = self.config.get("subject_name", "")

        # Generate descriptions for each image
        descriptions = []
        for img_path in sorted(image_paths):
            try:
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                output_file = os.path.join(output_dir, f"{base_filename}.txt")

                # Skip if description already exists and we're not overwriting
                if os.path.exists(output_file) and not self.config.get("overwrite_descriptions", False):
                    print(
                        f"Description for {base_filename} already exists, skipping")
                    with open(output_file, 'r') as f:
                        descriptions.append(f.read().strip())
                    continue

                # Generate description for this image
                description = self._generate_single_description(
                    img_path, prompt_template, subject_name)

                if description:
                    # Save the description
                    with open(output_file, 'w') as f:
                        f.write(description)
                    print(f"Generated description for {base_filename}")
                    descriptions.append(description)
                else:
                    print(
                        f"Failed to generate description for {base_filename}")

            except Exception as e:
                print(f"Error generating description for {img_path}: {e}")

        print(f"Generated {len(descriptions)} descriptions")
        return descriptions

    def _get_prompt_template(self, template_file: Optional[str] = None) -> str:
        """
        Get the prompt template from file or default.

        Args:
            template_file: Path to a text file containing the prompt template

        Returns:
            Prompt template as a string
        """
        default_template = """
Please create a single detailed sentence describing this image. 
The sentence should be appropriate for training a text-to-image model.
It should describe what's in the image in detail, including:

- The subject (person, object, etc.)
- The setting or background
- Any notable visual elements
- Lighting, style, and mood
- Any relevant poses, expressions, or actions

Write the description as a single detailed sentence with no additional commentary.
The description should be factual, clear, and objective.
"""

        # Use default if no file specified
        if not template_file:
            return default_template

        # Read from file if it exists
        if os.path.exists(template_file):
            try:
                with open(template_file, 'r') as f:
                    template = f.read()
                print(f"Using custom prompt template from {template_file}")
                return template
            except Exception as e:
                print(f"Error reading template file {template_file}: {e}")
                print("Using default template instead")
                return default_template
        else:
            print(
                f"Template file {template_file} not found, using default template")
            return default_template

    def _generate_single_description(self, image_path: str,
                                     prompt_template: str,
                                     subject_name: str = "") -> str:
        """
        Generate a description for a single image.

        Args:
            image_path: Path to the image file
            prompt_template: Template for the prompt to OpenAI
            subject_name: Name of the subject (if applicable)

        Returns:
            Generated description as a string
        """
        try:
            # Prepare the image for the API
            with open(image_path, "rb") as image_file:
                # Create the API request
                response = self.openai_client.chat.completions.create(
                    model=self.config.get(
                        "openai_model", "gpt-4-vision-preview"),
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert at describing images for text-to-image model training. {('The subject name is ' + subject_name + '.') if subject_name else ''}"
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )

                # Extract the description
                description = response.choices[0].message.content.strip()

                # Clean up the description
                description = self._clean_description(description)

                return description

        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return ""

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image as base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image data
        """
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _clean_description(self, description: str) -> str:
        """
        Clean up the generated description.

        Args:
            description: Raw description from OpenAI

        Returns:
            Cleaned description
        """
        # Remove quotation marks if present
        description = description.strip('"\'')

        # Remove any "Description:" prefix
        description = re.sub(r'^Description:\s*', '',
                             description, flags=re.IGNORECASE)

        # Ensure it ends with a period
        if not description.endswith('.'):
            description += '.'

        return description.strip()

    def analyze_descriptions(self, folder_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Analyze existing text files to extract patterns and themes.

        Args:
            folder_path: Directory containing text files (optional)

        Returns:
            Tuple of (analysis dictionary, list of descriptions)
        """
        folder_path = folder_path or self.config.get("processed_images_dir")

        # Get all text files
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        if not txt_files:
            print(f"No .txt files found in {folder_path}")
            return None, []

        print(f"Found {len(txt_files)} description files to analyze")

        # Read all descriptions
        descriptions = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                description = f.read().strip()
                descriptions.append(description)

        # Extract photo types, settings, poses, and attire
        photo_types, settings, poses, attire = [], [], [], []
        subject_name = self.config.get("subject_name")

        for desc in descriptions:
            # Extract photo type
            type_match = re.search(
                f'^([\\w\\s]+) of {subject_name}', desc, re.IGNORECASE) if subject_name else None
            if type_match:
                photo_types.append(type_match.group(1).strip().lower())

            # Extract setting/background
            setting_match = re.search(
                r'(against|in|at) ([\w\s]+)', desc, re.IGNORECASE)
            if setting_match:
                settings.append(setting_match.group(2).strip().lower())

            # Extract pose
            poses_keywords = ["sitting", "standing",
                              "smiling", "serious", "looking", "posing"]
            for keyword in poses_keywords:
                if keyword in desc.lower():
                    poses.append(keyword)

            # Extract attire
            attire_keywords = ["suit", "formal", "casual",
                               "shirt", "t-shirt", "jacket", "blazer"]
            for keyword in attire_keywords:
                if keyword in desc.lower():
                    attire.append(keyword)

        # Compile the analysis
        analysis = {
            "total_descriptions": len(descriptions),
            "common_photo_types": dict(Counter(photo_types).most_common(5)),
            "common_settings": dict(Counter(settings).most_common(5)),
            "common_poses": dict(Counter(poses).most_common(5)),
            "common_attire": dict(Counter(attire).most_common(5)),
            "sample_descriptions": random.sample(descriptions, min(5, len(descriptions)))
        }

        return analysis, descriptions

    def generate_new_prompts(self, analysis: Optional[Dict[str, Any]] = None,
                             descriptions: Optional[List[str]] = None,
                             num_prompts: Optional[int] = None,
                             system_template_file: Optional[str] = None,
                             user_template_file: Optional[str] = None) -> List[str]:
        """
        Generate new prompts based on analysis of existing descriptions.

        Args:
            analysis: Analysis dictionary (optional)
            descriptions: List of descriptions (optional)
            num_prompts: Number of prompts to generate (optional)
            system_template_file: Path to system prompt template file (optional)
            user_template_file: Path to user prompt template file (optional)

        Returns:
            List of generated prompts
        """
        if not self.openai_client:
            return ["OpenAI client not initialized. Check your API key."]

        if analysis is None:
            analysis, descriptions = self.analyze_descriptions()

        if not analysis:
            return ["No descriptions found to analyze."]

        num_prompts = num_prompts or self.config.get("num_prompts", 5)
        subject_name = self.config.get("subject_name", "")

        # Load system prompt from template file or use default
        system_prompt = self._load_template(
            system_template_file,
            "system_prompt_template.txt",
            default_template=self._get_default_system_prompt(
                num_prompts, subject_name)
        )

        # Load user prompt from template file or use default
        user_prompt = self._load_template(
            user_template_file,
            "user_prompt_template.txt",
            default_template=self._get_default_user_prompt(
                analysis, num_prompts)
        )

        # Format the templates with dynamic data
        system_prompt = system_prompt.format(
            num_prompts=num_prompts,
            subject_name=subject_name
        )

        # Create a dictionary of values for formatting
        format_values = {
            "num_prompts": num_prompts,
            "common_photo_types": analysis['common_photo_types'],
            "common_settings": analysis['common_settings'],
            "common_poses": analysis['common_poses'],
            "common_attire": analysis['common_attire'],
            "sample_descriptions": "\n".join(['- ' + desc for desc in analysis['sample_descriptions']])
        }

        # Format the user prompt with the analysis data
        user_prompt = user_prompt.format(**format_values)

        try:
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config.get("openai_model", "gpt-4-turbo-preview"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000
            )

            # Extract and format prompts
            content = response.choices[0].message.content
            prompt_lines = []

            for line in content.split('\n'):
                if line.strip().startswith("Prompt"):
                    # Remove the "Prompt X:" prefix
                    prompt_text = re.sub(r'^Prompt \d+:\s*', '', line).strip()
                    if prompt_text:
                        prompt_lines.append(prompt_text)

            return prompt_lines

        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [f"Error generating prompts: {e}"]

    def _load_template(self, template_file: Optional[str] = None,
                       default_filename: str = None,
                       default_template: str = None) -> str:
        """
        Load a template from a file, with fallbacks.

        Args:
            template_file: Direct path to template file
            default_filename: Filename to look for in templates directory
            default_template: Default template string to use if no file is found

        Returns:
            Template content as a string
        """
        # First try the direct template file path if provided
        if template_file and os.path.exists(template_file):
            try:
                with open(template_file, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading template file {template_file}: {e}")

        # Next, try to find the file in the templates directory
        if default_filename:
            # Look for the file in various possible template directories
            possible_paths = [
                # Check module's template directory
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "templates", default_filename),
                # Check current directory's template subdirectory
                os.path.join("templates", default_filename),
                # Check direct in current directory
                default_filename,
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            return f.read()
                    except Exception as e:
                        print(f"Error reading template file {path}: {e}")

        # Fall back to default template
        return default_template

    def _get_default_system_prompt(self, num_prompts: int, subject_name: str) -> str:
        """Returns the default system prompt template."""
        system_prompt = """You are an expert at creating diverse, detailed prompts for generating realistic images.
        
        Based on the analysis of existing image descriptions, create {num_prompts} new, unique prompts that:
        1. Each prompt should be diverse and creative
        2. Include interesting locations or settings
        3. Vary the photo types (selfie, portrait, full-body shot) across different prompts
        4. Include realistic lighting and weather conditions appropriate for each setting
        5. Maintain realism and feasibility
        6. Are detailed enough to generate high-quality images
        7. Format each prompt as a single detailed sentence

        Your prompts should be creative but consistent with the existing dataset patterns.
        """
        if subject_name:
            system_prompt += "\nThe subject of these images is a person named {subject_name}."

        return system_prompt

    def _get_default_user_prompt(self, analysis: Dict[str, Any], num_prompts: int) -> str:
        """Returns the default user prompt template."""
        return """
        Here's an analysis of existing descriptions:

        Common photo types: {common_photo_types}
        Common settings/backgrounds: {common_settings}
        Common poses: {common_poses}
        Common attire: {common_attire}

        Sample descriptions from the dataset:
        {sample_descriptions}

        Based on this analysis, generate {num_prompts} diverse new prompts for generating images.
        Each prompt should be different from the others.

        Some suggested settings or ideas:
        - Distinctive locations or environments
        - Different times of day and lighting conditions
        - Various moods and atmospheres
        - Different styles of photography
        - Seasonal variations

        Each prompt should be a single detailed sentence that would work well for a text-to-image model.
        Label them as "Prompt 1:", "Prompt 2:", etc.
        """

    def save_prompts(self, prompts: List[str], output_file: Optional[str] = None) -> str:
        """
        Save generated prompts to a text file.

        Args:
            prompts: List of prompts to save
            output_file: Path to save prompts (optional)

        Returns:
            Path to the output file
        """
        if output_file is None:
            output_dir = self.config.get("output_dir")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, "new_generation_prompts.txt")

        with open(output_file, 'w') as f:
            for i, prompt in enumerate(prompts, 1):
                f.write(f"Prompt {i}: {prompt}\n\n")

        print(f"Saved {len(prompts)} prompts to {output_file}")

        return output_file
