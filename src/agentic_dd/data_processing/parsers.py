'''
As of September 2026, the GWDG writes about their docling service:
'This service is currently in beta phase and is updated regularly. The same applies to the documentation.'
and
'This service is using a fork of the Docling API with many modifications, which will be published in the future.'

Behaviour and supported features should be expected to change, and as noted in 'normalizers.py' can be inconsistent/unexpected.
'''

import json
import os
from pathlib import Path
from typing import Dict, Any
import requests


def call_docling(file_path: str, api_url: str, api_key: str, timeout: int = 180) -> Dict[str, Any]:
    """Calls the Docling service for a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        with open(file_path, "rb") as file:
            response = requests.post(
                api_url,
                headers = headers,
                files = {"document": file},
                timeout = timeout
            )

    except requests.RequestException as e:
        raise RuntimeError(f"Docling API call failed: {e}") from e  # It is safer for the live system if this fails, rather than recovers

    except ValueError as e:
        raise RuntimeError(f"Invalid JSON from Docling API: {e}") from e

    return response