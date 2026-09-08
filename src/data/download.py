# src/data/download.py
import os
from pathlib import Path
from typing import Optional

import requests


def download_audio(
    url: str, filename: str, output_dir: str | Path, chunk_size=8192
) -> Optional[Path]:
    """
    Downloads an audio file from a given URL.

    Args:
        url: The URL of the audio file.
        filename: Desired filename.
        output_dir: Directory to save to.

    Returns:
        Full path to the downloaded file, or None if download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath: Path = output_dir / filename

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
