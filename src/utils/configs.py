# src/utils/configs.py

"""Configuration loading and resolution utilities."""

import math
import yaml
from pathlib import Path


def get_nested_config(config: dict, dotted_path: str, default=None):
    """
    Retrieve a value from a nested dict using a dotted path.

    Args:
        config: Root config dict
        dotted_path: Dot-separated key path, e.g. "training.mixed_precision.enabled"
        default: Value to return if path doesn't exist

    Returns:
        The value at the path, or default if not found.
    """
    keys = dotted_path.split(".")
    d = config
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def set_nested_config(config: dict, dotted_path: str, value):
    """
    Set a value in a nested dict using a dotted path.
    Creates intermediate dicts as needed.

    Args:
        config: Root config dict (modified in place)
        dotted_path: Dot-separated key path, e.g. "training.mixed_precision.enabled"
        value: Value to set
    """
    keys = dotted_path.split(".")
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def load_and_resolve_config(root_dir: str | Path, config_rel_path: str) -> dict:
    """
    Loads YAML from root_dir / config_rel_path, resolves all data paths to absolute,
    and computes segment_size from segment_seconds.

    Args:
        root_dir: Project root directory
        config_rel_path: Path to config.yaml relative to root_dir. e.g. "configs/config.yaml"

    Returns:
        Fully resolved config dict with absolute paths and derived audio params.
    """
    ROOT_DIR = Path(root_dir).resolve()
    full_config_path = ROOT_DIR / config_rel_path

    if not full_config_path.exists():
        raise FileNotFoundError(f"Config not found: {full_config_path}")

    with open(full_config_path, "r") as file:
        config = yaml.safe_load(file)

    config["project_root"] = str(ROOT_DIR)
    config["config_path"] = str(full_config_path)

    # 1. Resolve data paths to absolute
    if "data" not in config:
        raise KeyError("Missing 'data' section in config")

    for key in ["raw_audio_dir", "processed_npy_dir", "metadata_dir", "data_csv"]:
        if key not in config["data"]:
            raise KeyError(f"Missing 'data.{key}' in config")
        config["data"][key] = str(ROOT_DIR / config["data"][key])

    # 2. Compute segment_size - all keys required, no defaults
    if "audio" not in config or "model" not in config:
        raise KeyError("Missing 'audio' or 'model' section in config")

    audio_cfg = config["audio"]
    model_cfg = config["model"]

    for k in ["sr", "n_fft", "hop_length", "n_mels"]:
        if k not in audio_cfg:
            raise KeyError(f"Missing 'audio.{k}' in config")
    if "patch_size" not in model_cfg:
        raise KeyError(f"Missing 'model.patch_size' in config")

    if "segment_seconds" in audio_cfg:
        sr = audio_cfg["sr"]
        hop_length = audio_cfg["hop_length"]
        patch_size = model_cfg["patch_size"]
        segment_seconds = audio_cfg["segment_seconds"]

        frames = int(segment_seconds * sr / hop_length)
        segment_size = math.ceil(frames / patch_size) * patch_size

        audio_cfg["segment_size"] = segment_size
        audio_cfg["n_frames_raw"] = frames
        audio_cfg["segment_seconds_actual"] = segment_size * hop_length / sr

        print(
            f"✓ segment_seconds={segment_seconds}s -> {frames} frames -> segment_size={segment_size} "
            f"({segment_size * hop_length / sr:.2f}s) [patch={patch_size}]"
        )

    elif "segment_size" not in audio_cfg:
        raise KeyError(
            "config['audio'] must contain either 'segment_seconds' or 'segment_size'"
        )

    return config


def resolve_metadata_csv_path(config):
    """Return the metadata CSV path with EXACT audio configuration match."""
    data_cfg = config.get("data", {})
    metadata_dir = data_cfg.get("metadata_dir")

    if not metadata_dir:
        raise FileNotFoundError(
            "No metadata directory configured in config['data']['metadata_dir']"
        )

    metadata_dir_path = Path(metadata_dir)
    if not metadata_dir_path.is_dir():
        raise FileNotFoundError(
            f"Metadata directory does not exist: {metadata_dir_path}"
        )

    audio_cfg = config.get("audio", {})
    required_keys = ["sr", "n_fft", "hop_length", "n_mels"]
    missing = [k for k in required_keys if k not in audio_cfg]
    if missing:
        raise KeyError(f"Missing keys in config['audio']: {missing}")

    # Build exact expected filename signature
    signature = f"sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"

    # We expect the csv to CONTAIN this exact signature as a whole block
    expected_name_part = signature.lower()

    csv_files = sorted(metadata_dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV files found in {metadata_dir_path}")

    # Find exact match
    matches = [p for p in csv_files if expected_name_part in p.stem.lower()]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No metadata CSV found with EXACT signature: {signature}\n"
            f"Looked in: {metadata_dir_path}\n"
            f"Available: {[p.name for p in csv_files]}"
        )

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple CSVs found with signature {signature}. Be explicit.\n"
            f"Matches: {[p.name for p in matches]}"
        )

    return str(matches[0])
