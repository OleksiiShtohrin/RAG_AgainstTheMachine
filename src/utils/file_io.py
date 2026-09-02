"""Safe JSON file I/O operations with Pydantic model serialization."""

import json
import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


def read_json_file(
    file_path: str,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Safely read and parse a JSON file.

    Args:
        file_path: Target path to the JSON file.

    Returns:
        Parsed JSON data (dict or list), or None if file cannot be read.
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, (dict, list)) else None
    except Exception:
        return None


def save_pydantic_to_json(model: BaseModel, save_path: str) -> bool:
    """Save a Pydantic model instance to a formatted JSON file.

    Args:
        model: Pydantic model instance.
        save_path: Destination file path.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
        return True
    except Exception:
        return False
