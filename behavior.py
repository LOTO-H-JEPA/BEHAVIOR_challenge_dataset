import collections
import collections.abc
# Python 3.10+ removed collections.Iterator/Mapping/etc. — patch before streaming import
for _name in ("Iterator", "Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from logging import getLogger
from math import ceil
from tqdm import tqdm
import numpy as np
from streaming import MDSWriter
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from huggingface_hub import HfApi

logger = getLogger()


class BehaviorVideoDataset(torch.utils.data.Dataset):
    """BEHAVIOR dataset with deterministic episode-chunk sampling for pre-encoding/training."""

    def __init__(
        self,
        data_path,
        fpcs=16,
        fps=5,
        transform=None,
        camera_frame=False,
        state_start_idx=0,
        state_dim=7,
        action_dim=23,
        cache_parquet=False,
        cache_video_readers=False,
    ):
        """Initialize the dataset from a JSON manifest file.

        Args:
            data_path: Path to the JSON manifest listing all episodes.
            fpcs: Frames per clip (chunk size) used to partition each episode.
            fps: Target sampling rate in frames per second. If None, the native
                video FPS is used.
            transform: Optional callable applied to the raw uint8 video buffer
                returned by decord before it is stored in the sample dict.
            camera_frame: Reserved for future use; currently unused.
            state_start_idx: Column offset into the ``observation.state`` array
                from which ``state_dim`` values are sliced.
            state_dim: Number of state dimensions to keep after slicing.
            action_dim: Number of action dimensions to keep (leading columns).
            cache_parquet: If True, keep loaded DataFrames in memory to avoid
                repeated disk reads across workers.
            cache_video_readers: If True, keep ``VideoReader`` objects alive
                between calls to ``loadvideo_decord``.
        """
        self.data_path = data_path
        self.dataset_root = os.path.dirname(os.path.abspath(data_path))
        self.fpc = fpcs
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.state_start_idx = state_start_idx
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cache_parquet = cache_parquet
        self.cache_video_readers = cache_video_readers
        self._parquet_cache = {}
        self._video_reader_cache = {}

        manifest = self._load_manifest(data_path)
        self.samples = self._parse_samples(manifest)
        self.episode_plans = self._build_episode_plans()
        self.windows = self._build_window_index()

    def _load_manifest(self, manifest_path):
        """Load and return the JSON manifest as a dict.

        Args:
            manifest_path: Absolute or relative path to the manifest JSON file.

        Returns:
            Parsed manifest dictionary.
        """
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _parse_samples(self, manifest):
        """Parse episode entries from the manifest into path dicts.

        Args:
            manifest: Manifest dict as returned by ``_load_manifest``.

        Returns:
            List of dicts, each containing ``video_path`` and ``parquet_path``.

        Raises:
            ValueError: If no valid episodes are found in the manifest.
        """
        samples = []
        for ep in manifest.get("episodes", []):
            task_name = ep.get("task_name")
            episode_file = ep.get("episode_file")
            if task_name is None or episode_file is None:
                logger.warning(f"Skipping manifest entry missing task_name/episode_file: {ep}")
                continue
            episode_name = os.path.splitext(os.path.basename(episode_file))[0]
            base = os.path.join(self.dataset_root, task_name)
            video_path = os.path.join(base, "video", f"{episode_name}.mp4")
            parquet_path = os.path.join(base, "data", f"{episode_name}.parquet")
            samples.append({
                "video_path": video_path,
                "parquet_path": parquet_path,
            })

        if not samples:
            raise ValueError(f"No episodes found in manifest: {self.data_path}")

        return samples

    def _build_episode_plans(self):
        """Build per-episode sampling plans from ``self.samples``.

        Each plan records the sampled frame indices, the frame stride, and the
        usable episode length after reconciling video and parquet lengths.
        Episodes that are too short or raise an exception are skipped with a
        warning.

        Returns:
            List of plan dicts with keys ``sample_idx``, ``indices``,
            ``fstp``, and ``max_len``.

        Raises:
            ValueError: If no valid plans could be constructed.
        """
        plans = []
        for sample_idx, sample in enumerate(self.samples):
            try:
                indices, fstp, max_len = self._episode_sampled_indices(sample)
                if indices is None or len(indices) == 0:
                    logger.warning(f"Skipping sample due to insufficient frames: {sample}")
                    continue
                plans.append(
                    {
                        "sample_idx": sample_idx,
                        "indices": indices,
                        "fstp": fstp,
                        "max_len": max_len,
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping sample during episode planning sample={sample} {e=}")
        if not plans:
            raise ValueError(f"No valid episode plans found in manifest: {self.data_path}")
        logger.info(f"Built {len(plans)} valid episode plans")
        return plans

    def _episode_sampled_indices(self, sample):
        """Compute the frame-stride and sampled indices for a single episode.

        Opens the video briefly to read its FPS and length, then constructs a
        uniform index array at the configured target FPS.

        Args:
            sample: Dict with ``video_path`` and ``parquet_path`` keys.

        Returns:
            Tuple of ``(indices, fstp, max_len)`` where ``indices`` is a
            ``np.int64`` array of frame positions, ``fstp`` is the integer
            frame stride, and ``max_len`` is the usable episode length.
            Returns ``(None, fstp, max_len)`` when the episode is too short.

        Raises:
            ValueError: If the configured FPS is not positive.
        """
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        try:
            vfps = vr.get_avg_fps()
            vlen = len(vr)
        finally:
            del vr
        fps = self.fps if self.fps is not None else vfps
        if fps <= 0:
            raise ValueError(f"fps must be > 0. Got fps={fps} for {vpath}")
        fstp = max(1, ceil(vfps / fps))
        parquet_len = len(pd.read_parquet(ppath, columns=["action"]))
        if abs(vlen - parquet_len) > 2:
            logger.warning(f"Length mismatch {vpath}: video={vlen}, parquet={parquet_len}")
        max_len = min(vlen, parquet_len)
        if max_len < fstp:
            logger.warning(f"Too short episode {vpath}: max_len={max_len}, fstp={fstp}")
            return None, fstp, max_len
        indices = np.arange(0, max_len, fstp, dtype=np.int64)
        return indices, fstp, max_len

    def _build_window_index(self):
        """Build the flat list of (episode_idx, start_idx) window pairs.

        Each episode plan is partitioned into non-overlapping chunks of
        ``self.fpc`` sampled frames.  The resulting list is used directly as
        the dataset index.

        Returns:
            List of ``(episode_idx, start_idx)`` tuples.

        Raises:
            ValueError: If no valid windows are found across all plans.
        """
        windows = []
        for episode_idx, plan in enumerate(self.episode_plans):
            indices = plan["indices"]
            n = len(indices)
            if n == 0:
                logger.warning(f"Skipping episode_idx={episode_idx} with no sampled indices")
                continue
            for start in range(0, n, self.fpc):
                windows.append((episode_idx, start))
        if not windows:
            raise ValueError(f"No valid windows found in manifest: {self.data_path}")
        logger.info(
            f"Built BEHAVIOR window index with {len(windows)} non-overlapping windows "
            f"from {len(self.episode_plans)} valid episode plans"
        )
        return windows

    def __len__(self):
        """Return the total number of clip windows in the dataset."""
        return len(self.windows)

    def __getitem__(self, index):
        """Return the sample dict for a given window index.

        Retries up to 10 times with a randomly selected fallback window when
        loading fails, then re-raises on the final attempt.

        Args:
            index: Integer index into ``self.windows``.

        Returns:
            Dict with keys ``video``, ``actions``, ``states``,
            ``frame_indices``, ``episode_idx``, ``start_idx``, and
            ``valid_len``.

        Raises:
            Exception: Propagates the last loading exception after all retries
                are exhausted.
        """
        episode_idx, start_idx = self.windows[index]
        plan = self.episode_plans[episode_idx]
        sample = self.samples[plan["sample_idx"]]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                buffer, actions, states, indices = self.loadvideo_decord(sample, plan, start_idx=start_idx)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for sample={sample} {e=}")
                if attempt == max_retries - 1:
                    raise
                episode_idx, start_idx = self.windows[np.random.randint(self.__len__())]
                plan = self.episode_plans[episode_idx]
                sample = self.samples[plan["sample_idx"]]

        valid_len = min(self.fpc, len(plan["indices"]) - start_idx)
        return {
            "video": buffer,
            "actions": actions,
            "states": states,
            "frame_indices": indices,
            "episode_idx": episode_idx,
            "start_idx": start_idx,
            "valid_len": valid_len,
        }

    def loadvideo_decord(self, sample, plan, start_idx=0):
        """Load a clip window from disk and return frames, actions, and states.

        Reads the parquet file for states/actions and the corresponding video
        frames via decord.  Short windows at the end of an episode are
        right-padded with the last valid frame/action/state.

        Args:
            sample: Dict with ``video_path`` and ``parquet_path`` keys.
            plan: Episode plan dict produced by ``_build_episode_plans``,
                containing ``indices``, ``fstp``, and ``max_len``.
            start_idx: Offset (in sampled-index space) of the clip window
                within the episode.

        Returns:
            Tuple of ``(buffer, actions, states, window_indices)`` where
            ``buffer`` is a ``uint8`` ndarray of shape
            ``(fpc, H, W, C)``, ``actions`` and ``states`` are ``float32``
            ndarrays of shape ``(fpc, action_dim*fstp)`` and
            ``(fpc, state_dim)`` respectively, and ``window_indices`` is an
            ``int64`` ndarray of the actual video frame positions.

        Raises:
            ValueError: If required parquet columns are missing or array
                dimensions are too small for the configured slice parameters.
            RuntimeError: If the episode plan contains no valid indices.
        """
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]
        df = self._load_parquet(ppath)
        if "observation.state" not in df.columns or "action" not in df.columns:
            raise ValueError(f"Expected `observation.state` and `action` in parquet: {ppath}")
        full_states = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
        full_actions = np.asarray(df["action"].to_list(), dtype=np.float32)

        if full_actions.shape[1] < self.action_dim:
            raise ValueError(f"Action dim out of bounds for {ppath}: {full_actions.shape[1]=}, {self.action_dim=}")
        if full_states.shape[1] < self.state_start_idx + self.state_dim:
            raise ValueError(
                f"State slice out of bounds for {ppath}: {full_states.shape[1]=}, {self.state_start_idx=}, {self.state_dim=}"
            )

        states = full_states[:, self.state_start_idx : self.state_start_idx + self.state_dim]
        vr = self._get_video_reader(vpath)
        fstp = plan["fstp"]
        max_len = min(plan["max_len"], states.shape[0], full_actions.shape[0], len(vr))
        indices = plan["indices"]

        if len(indices) == 0:
            raise RuntimeError(f"No indices in episode plan for {vpath=}, {fstp=}, {max_len=}")

        end_idx = min(start_idx + self.fpc, len(indices))
        real_window_indices = indices[start_idx:end_idx]
        if len(real_window_indices) < self.fpc:
            pad = np.full((self.fpc - len(real_window_indices),), real_window_indices[-1], dtype=np.int64)
            window_indices = np.concatenate([real_window_indices, pad])
        else:
            window_indices = real_window_indices

        raw_states = states
        raw_actions = full_actions[:, : self.action_dim]
        states = []
        actions = []
        for i, start in enumerate(window_indices):
            start = int(start)
            if start >= max_len:
                states.append(np.zeros(self.state_dim, dtype=np.float32))
                actions.append(np.zeros(fstp * self.action_dim, dtype=np.float32))
                continue
            if i + 1 < len(real_window_indices):
                next_start = int(real_window_indices[i + 1])
            else:
                next_start = start + fstp
            end = min(max(next_start, start + 1), max_len)
            action_chunk = raw_actions[start:end]

            states.append(raw_states[start])

            if len(action_chunk) == 0:
                logger.warning(f"Empty action chunk for {vpath=}, {start=}, {end=}")
                action_chunk = np.zeros((fstp, self.action_dim), dtype=np.float32)
            elif len(action_chunk) < fstp:
                pad = np.repeat(action_chunk[-1:], fstp - len(action_chunk), axis=0)
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
            else:
                action_chunk = action_chunk[:fstp]

            actions.append(action_chunk.reshape(fstp * self.action_dim))

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        vr.seek(0)
        buffer = vr.get_batch(window_indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)
        return buffer, actions, states, window_indices

    def _load_parquet(self, ppath):
        """Load a parquet file, optionally from the in-memory cache.

        Args:
            ppath: Path to the parquet file.

        Returns:
            ``pandas.DataFrame`` for the requested file.
        """
        if not self.cache_parquet:
            return pd.read_parquet(ppath)
        cached = self._parquet_cache.get(ppath)
        if cached is not None:
            return cached
        df = pd.read_parquet(ppath)
        self._parquet_cache[ppath] = df
        return df

    def _get_video_reader(self, vpath):
        """Return a ``VideoReader`` for the given path, optionally cached.

        Args:
            vpath: Path to the MP4 video file.

        Returns:
            ``decord.VideoReader`` instance for the requested file.
        """
        if not self.cache_video_readers:
            return VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        cached = self._video_reader_cache.get(vpath)
        if cached is not None:
            return cached
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        self._video_reader_cache[vpath] = vr
        return vr


class _ShardUploader:
    """Uploads completed MDS shards to HF in a background thread.

    Polls write_dir every 30 s for new shard.*.mds files.  All but the last
    shard are considered complete and safe to upload (the last one may still
    be open by MDSWriter).  call stop_and_flush() after MDSWriter exits to
    drain remaining shards and index.json.

    When delete_local=True (HF-only mode), each shard is removed from disk
    immediately after a successful upload to keep disk usage bounded.
    """

    def __init__(self, write_dir, api, repo_id, path_in_repo, delete_local):
        """Initialize the uploader but do not start the background thread yet.

        Args:
            write_dir: Local directory that ``MDSWriter`` writes shards into.
            api: Authenticated ``huggingface_hub.HfApi`` instance.
            repo_id: HuggingFace dataset repository ID (e.g. ``"org/repo"``).
            path_in_repo: Prefix path inside the repository for all uploaded
                files.  Pass ``None`` or an empty string for the repo root.
            delete_local: If True, each shard is removed from disk immediately
                after a successful upload.
        """
        self._write_dir    = write_dir
        self._api          = api
        self._repo_id      = repo_id
        self._path_in_repo = path_in_repo
        self._delete_local = delete_local
        self._uploaded     = set()
        self._error        = None
        self._stop         = threading.Event()
        self._thread       = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Start the background polling thread."""
        self._thread.start()

    def stop_and_flush(self):
        """Signal the thread to stop, wait for it, then upload remaining files.

        Should be called after ``MDSWriter`` has exited so that the final shard
        and ``index.json`` are complete and safe to upload.

        Raises:
            RuntimeError: If the background thread encountered an unhandled
                exception during uploading.
        """
        self._stop.set()
        self._thread.join()
        if self._error:
            raise RuntimeError("Background shard upload failed") from self._error
        self._process(final=True)

    def _loop(self):
        """Background thread body: poll for new shards every 30 seconds."""
        try:
            while not self._stop.wait(timeout=30):
                self._process(final=False)
        except Exception as e:
            self._error = e

    def _process(self, final):
        """Upload any not-yet-uploaded shards (and optionally ``index.json``).

        When ``final`` is False the last shard is skipped because
        ``MDSWriter`` may still be writing to it.  When ``final`` is True all
        shards and ``index.json`` are included.  Each file is retried up to
        three times with back-off on failure.

        Args:
            final: If True, include the last shard and ``index.json``.
        """
        shards = sorted(glob.glob(os.path.join(self._write_dir, "shard.*.mds")))
        # Skip the last shard unless final — MDSWriter may still be writing to it.
        targets = shards if final else shards[:-1]
        if final:
            index = os.path.join(self._write_dir, "index.json")
            if os.path.exists(index):
                targets = targets + [index]
        for fpath in targets:
            if fpath in self._uploaded:
                continue
            fname   = os.path.basename(fpath)
            in_repo = f"{self._path_in_repo}/{fname}" if self._path_in_repo else fname
            for attempt in range(3):
                try:
                    self._api.upload_file(
                        path_or_fileobj=fpath,
                        path_in_repo=in_repo,
                        repo_id=self._repo_id,
                        repo_type="dataset",
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            self._uploaded.add(fpath)
            if self._delete_local and fpath.endswith(".mds"):
                os.remove(fpath)


class BehaviorEpisodePreencoder:
    """Run a vision encoder on BEHAVIOR clips and save pre-encoded episode shards."""

    def __init__(self, encoder, device=None, dtype=torch.float32):
        """Initialize the pre-encoder and move the encoder model to the target device.

        Args:
            encoder: A callable ``nn.Module`` that accepts a ``(B, C, T, H, W)``
                float tensor and returns either a tensor or a tuple whose first
                element is a tensor of shape ``(B, num_tokens, embed_dim)``.
            device: ``torch.device`` (or string) to run inference on.  Defaults
                to CUDA when available, otherwise CPU.
            dtype: Floating-point dtype used for encoder inputs and outputs.
                ``bfloat16`` outputs are cast to ``float16`` before being saved
                to disk.
        """
        self.encoder = encoder.eval()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.encoder.to(self.device)

    def _to_video_tensor(self, video):
        """Convert a raw video array/tensor to a ``(B, C, T, H, W)`` float tensor.

        Accepts numpy arrays or PyTorch tensors in any of the common layouts
        ``(B, T, H, W, C)``, ``(B, T, C, H, W)``, or ``(B, C, T, H, W)``.
        A missing batch dimension is added automatically for 4-D inputs.

        Args:
            video: ``np.ndarray`` or ``torch.Tensor`` representing a batch of
                video clips.

        Returns:
            ``torch.Tensor`` of shape ``(B, C, T, H, W)`` on ``self.device``
            with dtype ``self.dtype``.

        Raises:
            ValueError: If the input is not 4-D or 5-D, or if the channel axis
                cannot be inferred.
        """
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape={tuple(video.shape)}")
        # Normalize to [B, C, T, H, W] for 3D patch embedding.
        if video.shape[1] in (1, 3):
            # Already [B, C, T, H, W]
            pass
        elif video.shape[2] in (1, 3):
            # [B, T, C, H, W] -> [B, C, T, H, W]
            video = video.permute(0, 2, 1, 3, 4)
        elif video.shape[-1] in (1, 3):
            # [B, T, H, W, C] -> [B, C, T, H, W]
            video = video.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"Unable to infer channel axis for video shape={tuple(video.shape)}")
        return video.to(self.device, dtype=self.dtype, non_blocking=True)

    @staticmethod
    def behavior_preencode_collate(batch):
        """Collate a list of sample dicts into batched numpy arrays.

        Intended for use as the ``collate_fn`` argument of a
        ``torch.utils.data.DataLoader``.

        Args:
            batch: List of sample dicts as returned by
                ``BehaviorVideoDataset.__getitem__``.

        Returns:
            Dict mapping each key to a stacked numpy array with an added
            leading batch dimension.
        """
        return {
            "video": np.stack([item["video"] for item in batch], axis=0),
            "actions": np.stack([item["actions"] for item in batch], axis=0),
            "states": np.stack([item["states"] for item in batch], axis=0),
            "frame_indices": np.stack([item["frame_indices"] for item in batch], axis=0),
            "episode_idx": np.asarray([item["episode_idx"] for item in batch], dtype=np.int64),
            "start_idx": np.asarray([item["start_idx"] for item in batch], dtype=np.int64),
            "valid_len": np.asarray([item["valid_len"] for item in batch], dtype=np.int64),
        }

    @staticmethod
    def _gpu_status():
        """Return a short GPU utilisation string from ``nvidia-smi``.

        Returns:
            String of the form ``"gpu=<util>% mem=<used>/<total>MB"``, or
            ``"gpu=n/a"`` if ``nvidia-smi`` is unavailable.
        """
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()[0]
            util, mem_used, mem_total = [x.strip() for x in out.split(",")]
            return f"gpu={util}% mem={mem_used}/{mem_total}MB"
        except Exception:
            return "gpu=n/a"

    _MDS_COLUMNS = {
        "tokens":      "ndarray",
        "actions":     "ndarray",
        "states":      "ndarray",
        "frame_index": "int",
        "episode_idx": "int",
        "sample_idx":  "int",
        "frame_pos":   "int",
        "episode_len": "int",
    }

    @torch.inference_mode()
    def encode_full_episodes(self, dataset, output_dir=None, hf_repo_id=None, hf_path_prefix="", max_shard_bytes=1 << 30, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2):
        """Encode all episodes and write them as MDS shards.

        Iterates over ``dataset`` in window order, encodes each batch with the
        vision encoder, re-assembles the per-frame tokens in episode order, and
        writes complete episodes to MDS via ``MDSWriter``.  Optionally uploads
        finished shards to a HuggingFace dataset repository in the background
        while encoding continues.

        Args:
            dataset: ``BehaviorVideoDataset`` instance to encode.
            output_dir: Local directory to write MDS shards into.  Created if
                it does not exist.  Required unless ``hf_repo_id`` is given.
            hf_repo_id: HuggingFace dataset repository ID to upload shards to.
                When provided without ``output_dir``, shards are written to a
                temporary directory and deleted after upload.
            hf_path_prefix: Path prefix inside the HF repository for uploaded
                files.  Defaults to the repository root.
            max_shard_bytes: Maximum size in bytes for each MDS shard file.
                Defaults to 1 GiB.
            batch_size: Number of windows per data-loader batch.
            num_workers: Number of worker processes for the data loader.
            pin_memory: If True, use pinned memory in the data loader for
                faster host-to-device transfers.
            persistent_workers: If True, keep data-loader worker processes
                alive between batches.
            prefetch_factor: Number of batches to prefetch per worker.

        Raises:
            ValueError: If neither ``output_dir`` nor ``hf_repo_id`` is given.
            RuntimeError: If the background shard uploader fails.
        """
        if not output_dir and not hf_repo_id:
            raise ValueError("Either output_dir or hf_repo_id must be provided")

        tmp_dir = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            write_dir = output_dir
        else:
            tmp_dir = tempfile.mkdtemp()
            write_dir = tmp_dir

        data_loader = self._make_data_loader(dataset, batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor)
        state = {
            "encoded_episodes": 0,
            "encoded_frames": 0,
            "active_episode_idx": None,
            "active_buffer": None,
        }

        uploader = None
        if hf_repo_id:
            uploader = _ShardUploader(
                write_dir=write_dir,
                api=HfApi(),
                repo_id=hf_repo_id,
                path_in_repo=hf_path_prefix.strip("/") or None,
                delete_local=(tmp_dir is not None),
            )
            uploader.start()

        try:
            with MDSWriter(out=write_dir, columns=self._MDS_COLUMNS, size_limit=max_shard_bytes) as writer:
                pbar = tqdm(data_loader, desc="Encoding batches")
                for batch_idx, batch in enumerate(pbar):
                    if batch_idx % 10 == 0:
                        pbar.set_postfix_str(self._gpu_status())
                    tokens = self._encode_batch(batch["video"], dataset.fpc)
                    for b in range(tokens.shape[0]):
                        self._accumulate(state, dataset, writer, tokens, batch, b)
                self._flush_active_episode(state, dataset, writer)
            # MDSWriter context exit writes the final shard and index.json.
            if uploader:
                uploader.stop_and_flush()
                logger.info(f"Uploaded MDS shards to hf://{hf_repo_id}/{hf_path_prefix.strip('/') or ''}")
            logger.info(f"Pre-encoding finished: {state['encoded_episodes']} episodes, {state['encoded_frames']} frames")
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _flush_active_episode(self, state, dataset, writer):
        """Write the buffered windows for the current episode to the MDS writer.

        Windows are sorted by their ``start_idx`` before concatenation so that
        the written frame sequence is in chronological order regardless of
        data-loader ordering.  Clears the active buffer and increments the
        episode and frame counters in ``state`` afterwards.

        Args:
            state: Mutable state dict maintained by ``encode_full_episodes``,
                containing ``active_episode_idx``, ``active_buffer``,
                ``encoded_episodes``, and ``encoded_frames``.
            dataset: The ``BehaviorVideoDataset`` being encoded, used to look
                up the original ``sample_idx`` for each episode.
            writer: Open ``MDSWriter`` instance to write samples into.
        """
        if state["active_episode_idx"] is None or state["active_buffer"] is None or not state["active_buffer"]["tokens"]:
            return
        buf = state["active_buffer"]
        order = np.argsort(buf["starts"])
        episode_idx = int(state["active_episode_idx"])
        sample_idx  = int(dataset.episode_plans[episode_idx]["sample_idx"])
        tokens        = np.concatenate([buf["tokens"][i]        for i in order], axis=0)
        actions       = np.concatenate([buf["actions"][i]       for i in order], axis=0)
        states        = np.concatenate([buf["states"][i]        for i in order], axis=0)
        frame_indices = np.concatenate([buf["frame_indices"][i] for i in order], axis=0)
        T = tokens.shape[0]
        for t in range(T):
            writer.write({
                "tokens":      tokens[t],
                "actions":     actions[t],
                "states":      states[t],
                "frame_index": int(frame_indices[t]),
                "episode_idx": episode_idx,
                "sample_idx":  sample_idx,
                "frame_pos":   t,
                "episode_len": T,
            })
        state["encoded_episodes"] += 1
        state["encoded_frames"] += T
        state["active_episode_idx"] = None
        state["active_buffer"] = None

    def _make_data_loader(self, dataset, batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor):
        """Construct a non-shuffling DataLoader with the pre-encode collate function.

        Args:
            dataset: ``BehaviorVideoDataset`` to wrap.
            batch_size: Number of windows per batch.
            num_workers: Number of worker processes.
            pin_memory: Enable pinned memory for faster GPU transfers.
            persistent_workers: Keep worker processes alive between batches.
                Ignored when ``num_workers == 0``.
            prefetch_factor: Batches to prefetch per worker.  Ignored when
                ``num_workers == 0``.

        Returns:
            Configured ``torch.utils.data.DataLoader`` instance.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=self.behavior_preencode_collate,
        )

    def _encode_batch(self, video, fpc):
        """Encode a batch of video clips and reshape tokens per frame.

        Args:
            video: Raw video array of shape ``(B, T, H, W, C)`` or any layout
                accepted by ``_to_video_tensor``.
            fpc: Frames per clip; used to split the flat token sequence into
                per-frame groups.

        Returns:
            ``float32`` numpy array of shape
            ``(B, fpc, tokens_per_frame, embed_dim)``.

        Raises:
            AssertionError: If the total number of tokens is not divisible by
                ``fpc``.
        """
        tokens = self.encoder(self._to_video_tensor(video))
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        storage_dtype = torch.float16 if self.dtype == torch.bfloat16 else self.dtype
        tokens = tokens.detach().cpu().to(storage_dtype).numpy()
        assert tokens.shape[1] % fpc == 0, (
            f"Encoder output tokens ({tokens.shape[1]}) not divisible by fpc ({fpc})"
        )
        tokens_per_frame = tokens.shape[1] // fpc
        return tokens.reshape(tokens.shape[0], fpc, tokens_per_frame, tokens.shape[2])

    def _accumulate(self, state, dataset, writer, tokens, batch, b):
        """Accumulate encoded windows into the active episode buffer.

        When the episode index changes, the previous episode is flushed to the
        MDS writer before starting a new buffer.

        Args:
            state: Mutable state dict maintained by ``encode_full_episodes``.
            dataset: The source ``BehaviorVideoDataset``.
            writer: Open ``MDSWriter`` instance.
            tokens: Encoded token array of shape
                ``(B, fpc, tokens_per_frame, embed_dim)`` as returned by
                ``_encode_batch``.
            batch: Collated batch dict from the data loader.
            b: Batch index of the sample to accumulate.
        """
        episode_idx = int(batch["episode_idx"][b])
        valid_len   = int(batch["valid_len"][b])
        if state["active_episode_idx"] is None:
            state["active_episode_idx"] = episode_idx
            state["active_buffer"] = {"tokens": [], "actions": [], "states": [], "frame_indices": [], "starts": []}
        elif episode_idx != state["active_episode_idx"]:
            self._flush_active_episode(state, dataset, writer)
            state["active_episode_idx"] = episode_idx
            state["active_buffer"] = {"tokens": [], "actions": [], "states": [], "frame_indices": [], "starts": []}
        state["active_buffer"]["tokens"].append(tokens[b, :valid_len])
        state["active_buffer"]["actions"].append(batch["actions"][b, :valid_len])
        state["active_buffer"]["states"].append(batch["states"][b, :valid_len])
        state["active_buffer"]["frame_indices"].append(batch["frame_indices"][b, :valid_len])
        state["active_buffer"]["starts"].append(int(batch["start_idx"][b]))
