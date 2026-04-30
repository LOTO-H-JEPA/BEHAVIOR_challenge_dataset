import yaml
import os
import logging
from datasets import load_dataset
from huggingface_hub import hf_hub_download


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
    repo_id, file_path=None, use_hub_download=False, token=None, **kwargs
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
        logger.info(
            f"Downloading file '{file_path}' from repo '{repo_id}' via huggingface_hub."
        )
        return hf_hub_download(repo_id=repo_id, filename=file_path, token=token)
    else:
        logger.info(f"Loading dataset '{repo_id}' via datasets.load_dataset.")
        return load_dataset(repo_id, **kwargs)
