from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset_utils import load_from_huggingface, load_yaml_config, logger


class BaseDataset:
    """
    Base class for dataset handling. Provides common functionality for loading and managing datasets.
    """

    def __init__(self, use_hub_download: bool = False, token: str | None = None, **kwargs):
        """
        Initialize the BaseDataset using configuration from configs/dataset.yaml.
        Args:
            use_hub_download (bool, optional): Whether to use huggingface_hub for file download. Default is False.
            token (str, optional): Hugging Face authentication token.
            **kwargs: Additional arguments for dataset loading.
        """
        config = load_yaml_config("dataset.yaml")
        self.dataset_cfg = config.get("dataset", {})

        # Explicitly set each attribute from config (no fallbacks)
        self.camera_view_type = self.dataset_cfg["camera_view_type"]
        self.dataset_size = self.dataset_cfg["dataset_size"]
        self.eval_tasks = self.dataset_cfg["eval_tasks"]
        self.exclude_eval_tasks = self.dataset_cfg["exclude_eval_tasks"]
        self.include_eval_tasks_fully = self.dataset_cfg["include_eval_tasks_fully"]
        self.obs_resolution = self.dataset_cfg["obs_resolution"]
        self.fps = self.dataset_cfg["fps"]
        self.shard_size = self.dataset_cfg["shard_size"]
        self.base_dataset_destination = self.dataset_cfg["base_dataset_destination"]
        self.encoded_dataset_destination = self.dataset_cfg["encoded_dataset_destination"]
        self.encoded_dataset_destination_path = self.dataset_cfg[
            "encoded_dataset_destination_path"
        ]
        self.augmentation = self.dataset_cfg["augmentation"]
        self.encode_dataset = self.dataset_cfg["encode_dataset"]

        self.repo_id = "behavior-1k/2025-challenge-demos"
        self.use_hub_download = use_hub_download
        self.token = token
        self.kwargs = kwargs
        self.logger = logger

        config_preview = {
            "camera": self.camera_view_type,
            "size": self.dataset_size,
            "eval_tasks": len(self.eval_tasks),
            "exclude_eval_tasks": len(self.exclude_eval_tasks),
            "include_eval_tasks_fully": self.include_eval_tasks_fully,
            "obs_resolution": self.obs_resolution,
            "fps": self.fps,
            "shard_size": self.shard_size,
            "base_dst": self.base_dataset_destination,
            "encoded_dst": self.encoded_dataset_destination,
            "encode_dataset": self.encode_dataset,
        }
        self.logger.info(f"BaseDataset config: {config_preview}")

        # Build base dataset state immediately after initialization.
        self.meta = self.build_base_dataset()

    def _read_json_file(self, local_path: str | Path) -> Any:
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_jsonl_file(self, local_path: str | Path) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def build_base_dataset(self) -> dict[str, Any]:
        """
        Load required metadata files from the repository /meta folder.

        Returns:
            dict[str, Any]: dictionary containing parsed file contents keyed by filename.
        """
        required_meta_files = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
        loaded_meta: dict[str, Any] = {}

        self.logger.info("Building base dataset: loading required metadata files from /meta.")

        for file_path in required_meta_files:
            try:
                local_path = load_from_huggingface(
                    self.repo_id,
                    file_path=file_path,
                    use_hub_download=True,
                    token=self.token,
                    **self.kwargs,
                )

                if file_path.endswith(".json"):
                    parsed = self._read_json_file(local_path)
                else:
                    parsed = self._read_jsonl_file(local_path)

                loaded_meta[file_path] = parsed
            except Exception as exc:
                self.logger.warning(f"Missing or unreadable: {file_path} ({exc})")

        found_count = len(loaded_meta)
        total_count = len(required_meta_files)
        self.logger.info(
            f"Base dataset metadata loaded: {found_count}/{total_count} files found "
            f"({', '.join(sorted(loaded_meta.keys()))})"
        )

        return loaded_meta
