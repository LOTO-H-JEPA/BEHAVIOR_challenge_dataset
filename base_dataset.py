from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .dataset_utils import (
    episode_duration_minutes,
    episode_task_name,
    load_from_huggingface,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    load_yaml_config,
    logger,
)


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
        self.seed = self.dataset_cfg["seed"]
        self.eval_tasks = self.dataset_cfg["eval_tasks"]
        self.exclude_eval_tasks = self.dataset_cfg["exclude_eval_tasks"]
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
            "seed": self.seed,
            "eval_tasks": len(self.eval_tasks),
            "exclude_eval_tasks": self.exclude_eval_tasks,
            "obs_resolution": self.obs_resolution,
            "fps": self.fps,
            "shard_size": self.shard_size,
            "base_dst": self.base_dataset_destination,
            "encoded_dst": self.encoded_dataset_destination,
            "encode_dataset": self.encode_dataset,
        }
        self.logger.info(f"BaseDataset config: {config_preview}")

        # Build base dataset state immediately after initialization.
        self.build_base_dataset()

    def build_base_dataset(self) -> dict[str, Any]:
        """
        Load required metadata files from the repository /meta folder.

        Returns:
            dict[str, Any]: dictionary containing parsed file contents keyed by filename.
        """
        required_meta_files = {
            "info": "meta/info.json",
            "tasks": "meta/tasks.jsonl",
            "episodes": "meta/episodes.jsonl",
        }
        loaded_meta: dict[str, Any] = {"info": None, "tasks": None, "episodes": None}

        self.logger.info("Building base dataset: loading required metadata files from /meta.")

        for key, file_path in required_meta_files.items():
            try:
                local_path = load_from_huggingface(
                    self.repo_id,
                    file_path=file_path,
                    use_hub_download=True,
                    token=self.token,
                    **self.kwargs,
                )

                if file_path.endswith(".json"):
                    parsed = load_json_file(local_path)
                else:
                    parsed = load_jsonl_file(local_path)

                loaded_meta[key] = parsed
            except Exception as exc:
                self.logger.warning(f"Missing or unreadable: {file_path} ({exc})")

        self.info = loaded_meta["info"]
        self.tasks = loaded_meta["tasks"]
        self.episodes = loaded_meta["episodes"]

        found_count = len([v for v in loaded_meta.values() if v is not None])
        total_count = len(required_meta_files)
        self.logger.info(
            f"Base dataset metadata loaded: {found_count}/{total_count} files found "
        )

        if found_count == total_count:
            self.selected_meta = self._build_selected_meta()
            self._log_metadata_preview()
        else:
            self.selected_meta = None
            self.logger.warning(
                "Not all required metadata files were loaded successfully; skipping episode selection."
            )

        return loaded_meta

    def _build_selected_meta(self) -> dict[str, Any]:
        rng = random.Random(self.seed)
        target_hours = float(self.dataset_size)
        eval_task_set = set(self.eval_tasks)

        fps = float(self.info.get("fps", 30.0))
        total_tasks = int(self.info.get("total_tasks", 50))
        per_task_budget_hours = target_hours / float(total_tasks)
        episode_index_lookup, episodes_by_task_desc, task_lookup = self._build_episode_task_lookups()
        self._log_task_episode_link_summary(task_lookup, episodes_by_task_desc)

        selected: list[dict[str, Any]] = []
        selected_hours = 0.0

        for task_desc, task_meta in task_lookup.items():
            task_name = task_meta.get("task_name")
            if self.exclude_eval_tasks and task_name in eval_task_set:
                continue

            remaining_budget_hours = per_task_budget_hours
            episode_ids = episodes_by_task_desc.get(task_desc, []).copy()

            while episode_ids and remaining_budget_hours > 0:
                sampled_id = rng.choice(episode_ids)
                episode_ids.remove(sampled_id)
                episode = episode_index_lookup.get(sampled_id)
                if not episode:
                    continue

                duration_minutes = episode_duration_minutes(episode, fps)
                duration_hours = duration_minutes / 60.0
                if duration_hours <= 0:
                    continue

                selected.append(
                    self._build_selected_episode_entry(
                        task_desc=task_desc,
                        task_name=task_name,
                        task_index=task_meta.get("task_index"),
                        sampled_id=sampled_id,
                        duration_minutes=duration_minutes,
                        episode=episode,
                    )
                )
                remaining_budget_hours -= duration_hours
                selected_hours += duration_hours

        selected_meta = {
            "target_hours": target_hours,
            "selected_hours": selected_hours,
            "per_task_budget_hours": per_task_budget_hours,
            "num_selected_episodes": len(selected),
            "exclude_eval_tasks": self.exclude_eval_tasks,
            "eval_tasks": list(eval_task_set),
            "episodes": selected,
        }
        self.logger.info(
            f"Selected {len(selected)} episodes totaling {selected_hours:.3f}h "
            f"(target={target_hours:.3f}h, per_task_budget={per_task_budget_hours:.3f}h)."
        )
        output_dir = self.base_dataset_destination
        if not output_dir or str(output_dir).lower() in {"none", "null"}:
            output_dir = "output"

        metadata_file = Path(output_dir) / "meta.json"
        save_json_file(str(metadata_file), selected_meta)
        self.logger.info(f"Saved selected metadata to {metadata_file.resolve()}")
        return selected_meta
   
    def _log_metadata_preview(self) -> None:
        tasks_count = len(self.tasks) if isinstance(self.tasks, list) else 0
        episodes_count = len(self.episodes) if isinstance(self.episodes, list) else 0
        selected_episodes = (
            self.selected_meta.get("episodes", []) if isinstance(self.selected_meta, dict) else []
        )
        selected_count = len(selected_episodes) if isinstance(selected_episodes, list) else 0

        task_totals: dict[str, int] = {}
        task_duration_totals: dict[str, float] = {}
        fps = float(self.info.get("fps", 30.0))
        for episode in selected_episodes if isinstance(selected_episodes, list) else []:
            task_name = str(episode_task_name(episode))
            task_totals[task_name] = task_totals.get(task_name, 0) + 1
            duration_minutes = episode_duration_minutes(episode, fps)
            task_duration_totals[task_name] = (
                task_duration_totals.get(task_name, 0.0) + duration_minutes
            )

        total_duration_minutes = sum(task_duration_totals.values())

        contribution_summary = "none"
        if selected_count > 0 and task_totals:
            sorted_totals = sorted(task_totals.items(), key=lambda item: (-item[1], item[0]))
            contribution_parts: list[str] = []
            for task, count in sorted_totals:
                duration_share_pct = 0.0
                if total_duration_minutes > 0:
                    duration_share_pct = (
                        task_duration_totals.get(task, 0.0) / total_duration_minutes
                    ) * 100
                contribution_parts.append(f"{task}: {count} eps ({duration_share_pct:.1f}%)")
            contribution_summary = ", ".join(contribution_parts)

        self.logger.info("Metadata preview:")
        self.logger.info(
            f"  counts: tasks={tasks_count}, episodes={episodes_count}, selected:episodes={selected_count}"
        )
        self.logger.info(f"  task_contributions: [{contribution_summary}]")


    def _build_episode_task_lookups(
        self,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, list[int]], dict[str, dict[str, Any]]]:
        task_lookup: dict[str, dict[str, Any]] = {}
        for task in self.tasks:
            description = task.get("task")
            if description:
                task_lookup[description] = {
                    "task_index": task.get("task_index"),
                    "task_name": task.get("task_name"),
                    "task": description,
                }
        expected_total_tasks = int(self.info.get("total_tasks", 50))
        if len(task_lookup) != expected_total_tasks:
            raise ValueError(
                f"Expected {expected_total_tasks} tasks in tasks metadata, found {len(task_lookup)}."
            )

        episode_index_lookup: dict[int, dict[str, Any]] = {}
        episodes_by_task_desc: dict[str, list[int]] = {}
        for episode in self.episodes:
            episode_id = episode.get("episode_index")
            if episode_id is None:
                self.logger.warning("Encountered episode entry without episode_index; skipping.")
                continue
            episode_index_lookup[int(episode_id)] = episode
            for task_desc in (episode.get("tasks") or []):
                if task_desc in task_lookup:
                    episodes_by_task_desc.setdefault(task_desc, []).append(int(episode_id))

        return episode_index_lookup, episodes_by_task_desc, task_lookup

    def _log_task_episode_link_summary(
        self, task_lookup: dict[str, dict[str, Any]], episodes_by_task_desc: dict[str, list[int]]
    ) -> None:
        task_episode_counts = {
            task_desc: len(episodes_by_task_desc.get(task_desc, []))
            for task_desc in task_lookup
        }
        total_task_episode_links = sum(task_episode_counts.values())
        self.logger.info(
            f"Found episode links for {len(task_episode_counts)}/{len(task_lookup)} tasks; "
            f"total task-episode links={total_task_episode_links}."
        )

    def _build_selected_episode_entry(
        self,
        task_desc: str,
        task_name: str | None,
        task_index: int | None,
        sampled_id: int,
        duration_minutes: float,
        episode: dict[str, Any],
    ) -> dict[str, Any]:
        episode_chunk = sampled_id // int(self.info["chunks_size"])
        path_vars = {"episode_chunk": episode_chunk, "episode_index": sampled_id}
        data_parquet_file = self.info["data_path"].format(**path_vars)
        episode_file = self.info["metainfo_path"].format(**path_vars)

        if self.camera_view_type in {"all"}:
            video_keys = [
                "observation.images.rgb.head",
                "observation.images.rgb.left_wrist",
                "observation.images.rgb.right_wrist",
            ]
        elif self.camera_view_type in {"head", "left_wrist", "right_wrist"}:
            video_keys = [f"observation.images.rgb.{self.camera_view_type}"]
        else:
            raise ValueError(
                f"Unsupported camera_view_type '{self.camera_view_type}'. "
                "Expected one of: all, head, left_wrist, right_wrist."
            )
        video_files = [
            self.info["video_path"].format(**(path_vars | {"video_key": video_key}))
            for video_key in video_keys
        ]

        return {
            "task": task_desc,
            "task_name": task_name,
            "task_index": task_index,
            "episode_index": sampled_id,
            "duration_minutes": duration_minutes,
            "video_file": video_files[0],
            "video_files": video_files,
            "data_parquet_file": data_parquet_file,
            "episode_file": episode_file,
            "raw": episode,
        }
