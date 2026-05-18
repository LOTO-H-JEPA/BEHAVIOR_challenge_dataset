from __future__ import annotations

import concurrent.futures
import random
import re
import time
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from tqdm import tqdm

from base_dataset_utils import (
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
        config = load_yaml_config("base_dataset.yaml")
        self.dataset_cfg = config.get("dataset", {})

        # Explicitly set each attribute from config (no fallbacks)
        self.camera_view_type = self.dataset_cfg["camera_view_type"]
        self.dataset_size = self.dataset_cfg["dataset_size"]
        self.seed = self.dataset_cfg["seed"]
        self.eval_tasks = self.dataset_cfg["eval_tasks"]
        self.exclude_eval_tasks = self.dataset_cfg["exclude_eval_tasks"]
        self.base_dataset_destination = self.dataset_cfg["base_dataset_destination"]
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
            "base_dst": self.base_dataset_destination,
        }
        self.logger.info(f"BaseDataset config: {config_preview}")
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
        """Sample episodes per task and persist selected metadata to disk."""
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

        metadata_file = Path(output_dir) / "manifest.json"
        save_json_file(str(metadata_file), selected_meta)
        self.logger.info(f"Saved selected metadata to {metadata_file.resolve()}")
        self._download_selected_episodes(selected_meta)
        return selected_meta
     
    def _download_selected_episodes(self, selected_meta: dict[str, Any]) -> None:
        """
        Download all files referenced by the selected manifest and store them
        in a LeRobot-like directory layout per task:
          <base_dataset_destination>/<task>/data
          <base_dataset_destination>/<task>/video
          <base_dataset_destination>/<task>/meta
        """
        episodes = selected_meta.get("episodes", [])
        if not episodes:
            self.logger.warning("No selected episodes available; skipping downloads.")
            return

        output_dir = Path(self.base_dataset_destination or "output")
        files_to_download = self._build_download_targets(episodes, output_dir)

        file_sizes_by_path = self._fetch_remote_file_sizes()
        estimated_total_bytes = sum(file_sizes_by_path.get(path, 0) for _, path, _ in files_to_download)
        has_size_estimate = estimated_total_bytes > 0
        progress = self._create_global_download_progress(
            total_bytes=estimated_total_bytes,
            total_files=len(files_to_download),
            has_size_estimate=has_size_estimate,
        )

        def _download_one(args):
            """Download a single remote file and copy it to the local directory.

            Args:
                args: Tuple of ``(local_dir, remote_path, category)`` where
                    ``local_dir`` is the destination ``Path``, ``remote_path``
                    is the relative repo path, and ``category`` is a label
                    (e.g. ``"data"``, ``"video"``, ``"meta"``) used in
                    warning messages.

            Returns:
                Tuple of ``(file_size_bytes, exception_or_None)``.
            """
            local_dir, remote_path, category = args
            try:
                downloaded_path = load_from_huggingface(
                    self.repo_id,
                    file_path=remote_path,
                    use_hub_download=True,
                    token=self.token,
                    **self.kwargs,
                )
                local_dir.mkdir(parents=True, exist_ok=True)
                destination_path = local_dir / Path(remote_path).name
                shutil.copy2(downloaded_path, destination_path)
                file_size = destination_path.stat().st_size
                progress.update(file_sizes_by_path.get(remote_path, file_size) if has_size_estimate else 1)
                return file_size, None
            except Exception as exc:
                self.logger.warning(f"Failed to download [{category}] {remote_path}: {exc}")
                if not has_size_estimate:
                    progress.update(1)
                return 0, exc

        num_workers = min(8, len(files_to_download))
        download_start = time.time()
        previous_disable_pb = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        disable_progress_bars()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_download_one, files_to_download))
        finally:
            progress.close()
            enable_progress_bars()
            if previous_disable_pb is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_disable_pb

        bytes_downloaded = sum(r[0] for r in results)
        missing_files = sum(1 for r in results if r[1] is not None)
        found_files = len(results) - missing_files
        elapsed_seconds = time.time() - download_start
        gib_downloaded = bytes_downloaded / (1024 ** 3)
        total_files = len(files_to_download)

        self.logger.info("Download summary")
        self.logger.info(f"  files found: {found_files}/{total_files} (all_found={found_files == total_files})")
        self.logger.info(f"  missing files: {missing_files}")
        self.logger.info(f"  downloaded size: {gib_downloaded:.3f} GiB")
        self.logger.info(f"  download time: {elapsed_seconds:.2f} seconds")

    def _build_download_targets(
        self,
        episodes: list[dict[str, Any]],
        output_dir: Path,
    ) -> list[tuple[Path, str, str]]:
        """Build local/remote/category download targets for selected episodes."""
        files_to_download: list[tuple[Path, str, str]] = []
        for episode in episodes:
            task_name = str(episode.get("task_name") or episode.get("task") or "unknown_task")
            safe_task_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", task_name).strip("_") or "unknown_task"
            task_root = output_dir / safe_task_name

            data_rel = episode.get("data_parquet_file")
            if data_rel:
                files_to_download.append((task_root / "data", str(data_rel), "data"))

            episode_rel = episode.get("episode_file")
            if episode_rel:
                files_to_download.append((task_root / "meta", str(episode_rel), "meta"))

            if self.camera_view_type == "multi":
                view_names = ["head", "left_wrist", "right_wrist"]
            else:
                view_names = ["head"]
            for view_key, video_rel in zip(view_names, episode.get("video_files", [])):
                files_to_download.append((task_root / "video" / view_key, str(video_rel), "video"))
        return files_to_download

    def _fetch_remote_file_sizes(self) -> dict[str, int]:
        """Fetch remote dataset file sizes keyed by relative repo path."""
        file_sizes_by_path: dict[str, int] = {}
        try:
            repo_info = HfApi(token=self.token).repo_info(
                repo_id=self.repo_id,
                repo_type="dataset",
                files_metadata=True,
            )
            for sibling in getattr(repo_info, "siblings", []) or []:
                sibling_path = getattr(sibling, "rfilename", None)
                sibling_size = getattr(sibling, "size", None)
                if sibling_path and isinstance(sibling_size, int):
                    file_sizes_by_path[str(sibling_path)] = sibling_size
        except Exception as exc:
            self.logger.warning(f"Could not fetch remote file sizes for global ETA: {exc}")
        return file_sizes_by_path

    def _create_global_download_progress(
        self,
        total_bytes: int,
        total_files: int,
        has_size_estimate: bool,
    ) -> tqdm:
        """Create the single global tqdm progress bar for selected-file downloads."""
        progress_unit = "B" if has_size_estimate else "file"
        progress_total = total_bytes if has_size_estimate else total_files
        return tqdm(
            total=progress_total,
            desc="Downloading selected episode files (global)",
            unit=progress_unit,
            unit_scale=has_size_estimate,
            unit_divisor=1024,
        )
   
    def _log_metadata_preview(self) -> None:
        """Log selected-episode counts and per-task duration-weighted contributions."""
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
        """Build lookup maps for episodes and task metadata relationships."""
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
        """Log high-level statistics for task-to-episode link coverage."""
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
        """Construct a normalized selected-episode metadata record."""
        episode_chunk = sampled_id // int(self.info["chunks_size"])
        path_vars = {"episode_chunk": episode_chunk, "episode_index": sampled_id}
        data_parquet_file = self.info["data_path"].format(**path_vars)
        episode_file = self.info["metainfo_path"].format(**path_vars)

        if self.camera_view_type == "multi":
            video_keys = [
                "observation.images.rgb.head",
                "observation.images.rgb.left_wrist",
                "observation.images.rgb.right_wrist",
            ]
        elif self.camera_view_type == "head":
            video_keys = ["observation.images.rgb.head"]
        else:
            raise ValueError(
                f"Unsupported camera_view_type '{self.camera_view_type}'. "
                "Expected one of: head, multi."
            )
        video_files = [
            self.info["video_path"].format(**(path_vars | {"video_key": video_key}))
            for video_key in video_keys
        ]

        # Sanitize task_name to match the directory created by _build_download_targets.
        import re as _re
        safe_task_name = _re.sub(r"[^a-zA-Z0-9._-]+", "_", task_name or "unknown_task").strip("_") or "unknown_task"
        return {
            "task": task_desc,
            "task_name": safe_task_name,
            "task_index": task_index,
            "episode_index": sampled_id,
            "duration_minutes": duration_minutes,
            "video_file": video_files[0],
            "video_files": video_files,
            "data_parquet_file": data_parquet_file,
            "episode_file": episode_file,
            "raw": episode,
        }

if __name__ == "__main__":
    BaseDataset()