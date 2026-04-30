import yaml
import os
import logging
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


# --- Logging setup: module-level logger with console handler and formatter ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

def load_yaml_config(config_filename):
    """
    Load a YAML configuration file from the configs directory.
    Args:
        config_filename (str): The filename of the YAML config (e.g., 'dataset.yaml').
    Returns:
        dict: Parsed YAML config, or empty dict if not found or error.
    """
    config_path = os.path.join(os.path.dirname(__file__), "configs", config_filename)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}")
        return {}


def load_from_huggingface(
    repo_id,
    file_path=None,
    use_hub_download=False,
    token=None,
    download_mode="single_file",
    **kwargs,
):
    """
    Load a dataset or file from a Hugging Face dataset repository.

    Args:
        repo_id (str): The Hugging Face dataset repo name (e.g., 'user/dataset').
        file_path (str, optional): Path to a specific file in the repo. If None, loads the dataset.
        use_hub_download (bool): If True, download a specific file using huggingface_hub. If False, use datasets.load_dataset.
        token (str, optional): Hugging Face authentication token for private repos.
        **kwargs: Additional arguments for load_dataset.
    Returns:
        Loaded dataset or local file path.
    """
    if use_hub_download and file_path:
        if download_mode == "snapshot":
            return snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                allow_patterns=file_path,
                **kwargs,
            )
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_path,
            token=token,
            **kwargs,
        )
    return load_dataset(repo_id, **kwargs)


def load_json_file(file_path):
    """Load and return JSON content from a UTF-8 encoded file path."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(file_path):
    """Load a JSONL file and return a list of parsed JSON objects."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json_file(file_path, data):
    """Save JSON-serializable data to disk, creating parent directories as needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def episode_task_name(episode):
    """Return the best available task label for an episode entry."""
    return (
        episode.get("task_name")
        or episode.get("task")
        or episode.get("task_id")
        or "unknown_task"
    )


def episode_duration_minutes(episode, fps=None):
    """Return an episode duration in minutes from available duration fields."""
    if "duration_minutes" in episode:
        return float(episode["duration_minutes"])
    if "duration_seconds" in episode:
        return float(episode["duration_seconds"]) / 60.0
    if "duration_hours" in episode:
        return float(episode["duration_hours"]) * 60.0
    if "length" in episode and fps:
        return (float(episode["length"]) / float(fps)) / 60.0
    if "num_frames" in episode and fps:
        return float(episode["num_frames"]) / float(fps) / 60.0
    return 0.0
