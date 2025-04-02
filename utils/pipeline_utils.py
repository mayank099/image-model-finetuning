"""
utils/pipeline_utils.py
Utility functions for pipeline operations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_env_file():
    """Load environment variables from .env file."""
    if os.path.isfile(os.path.join(os.getcwd(), '.env')):
        load_dotenv(os.path.join(os.getcwd(), '.env'))
    else:
        project_root = Path(__file__).parent
        while not (project_root / '.env').exists():
            if project_root.parent == project_root:
                raise FileNotFoundError(
                    ".env file not found in any parent directory")
            project_root = project_root.parent
        load_dotenv(project_root / '.env')
        print(f"Loaded environment variables from {project_root / '.env'}")
