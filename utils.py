# this is the file that will be used to contain utility functions for the agents
import boto3
import json
import time
from datetime import datetime
from boto3.session import Session
import os
import time
import logging
import yaml
from pathlib import Path
from textwrap import wrap
from typing import Optional, Dict, Union, List, Tuple, Any

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(config_file: Union[Path, str]) -> Optional[Dict]:
    """
    Load configuration from a local file.

    :param config_file: Path to the local file
    :return: Dictionary with the loaded configuration
    """
    try:
        config_data: Optional[Dict] = None
        logger.info(f"Loading config from local file system: {config_file}")
        content = Path(config_file).read_text()
        config_data = yaml.safe_load(content)
        logger.info(f"Loaded config from local file system: {config_data}")
    except Exception as e:
        logger.error(f"Error loading config from local file system: {e}")
        config_data = None
    return config_data
