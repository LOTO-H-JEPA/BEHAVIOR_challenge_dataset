import concurrent.futures
import io
import json
import os
import subprocess
import tarfile
from logging import getLogger
from math import ceil
from tqdm import tqdm
import numpy as np
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
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _parse_samples(self, manifest):
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
        return len(self.windows)

    def __getitem__(self, index):
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
        if not self.cache_parquet:
            return pd.read_parquet(ppath)
        cached = self._parquet_cache.get(ppath)
        if cached is not None:
            return cached
        df = pd.read_parquet(ppath)
        self._parquet_cache[ppath] = df
        return df

    def _get_video_reader(self, vpath):
        if not self.cache_video_readers:
            return VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        cached = self._video_reader_cache.get(vpath)
        if cached is not None:
            return cached
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        self._video_reader_cache[vpath] = vr
        return vr


class BehaviorEpisodePreencoder:
    """Run a vision encoder on BEHAVIOR clips and save pre-encoded episode shards."""

    def __init__(self, encoder, device=None, dtype=torch.float32):
        self.encoder = encoder.eval()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.encoder.to(self.device)

    def _to_video_tensor(self, video):
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

    @torch.inference_mode()
    def encode_full_episodes(self, dataset, output_dir=None, hf_repo_id=None, hf_path_prefix="", max_shard_bytes=1 << 30, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if not output_dir and not hf_repo_id:
            raise ValueError("Either output_dir or hf_repo_id must be provided")

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=self.behavior_preencode_collate,
        )
        state = {
            "shard_data": [], "shard_id": 0, "shard_bytes": 0,
            "encoded_episodes": 0, "total_shards_written": 0,
            "active_episode_idx": None, "active_buffer": None,
            "pending_write": None,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as write_executor:
            pbar = tqdm(data_loader, desc="Loading batches")
            for batch_idx, batch in enumerate(pbar):
                if batch_idx % 10 == 0:
                    pbar.set_postfix_str(self._gpu_status())
                tokens = self.encoder(self._to_video_tensor(batch["video"]))
                if isinstance(tokens, (tuple, list)):
                    tokens = tokens[0]
                storage_dtype = torch.float16 if self.dtype == torch.bfloat16 else self.dtype
                tokens = tokens.detach().cpu().to(storage_dtype).numpy()
                # Reshape [B, N, D] -> [B, fpc, patches_per_frame, D] so axis 1 aligns with frames.
                assert tokens.shape[1] % dataset.fpc == 0, (
                    f"Encoder output tokens ({tokens.shape[1]}) not divisible by fpc ({dataset.fpc})"
                )
                tokens_per_frame = tokens.shape[1] // dataset.fpc
                tokens = tokens.reshape(tokens.shape[0], dataset.fpc, tokens_per_frame, tokens.shape[2])
                for b in range(tokens.shape[0]):
                    episode_idx = int(batch["episode_idx"][b])
                    valid_len = int(batch["valid_len"][b])
                    if state["active_episode_idx"] is None:
                        state["active_episode_idx"] = episode_idx
                        state["active_buffer"] = {"tokens": [], "actions": [], "states": [], "frame_indices": [], "starts": []}
                    elif episode_idx != state["active_episode_idx"]:
                        self._flush_active_episode(state, dataset, output_dir, hf_repo_id, hf_path_prefix, max_shard_bytes, write_executor)
                        state["active_episode_idx"] = episode_idx
                        state["active_buffer"] = {"tokens": [], "actions": [], "states": [], "frame_indices": [], "starts": []}

                    state["active_buffer"]["tokens"].append(tokens[b, :valid_len])
                    state["active_buffer"]["actions"].append(batch["actions"][b, :valid_len])
                    state["active_buffer"]["states"].append(batch["states"][b, :valid_len])
                    state["active_buffer"]["frame_indices"].append(batch["frame_indices"][b, :valid_len])
                    state["active_buffer"]["starts"].append(int(batch["start_idx"][b]))

            self._flush_active_episode(state, dataset, output_dir, hf_repo_id, hf_path_prefix, max_shard_bytes, write_executor)

            if state["pending_write"] is not None:
                state["pending_write"].result()

            if state["shard_data"]:
                self._write_shard(output_dir=output_dir, hf_repo_id=hf_repo_id, hf_path_prefix=hf_path_prefix, shard_id=state["shard_id"], shard_data=state["shard_data"])
                state["total_shards_written"] += 1

        if output_dir and hf_repo_id:
            index_path = os.path.join(output_dir, "shard_index.json")
            if os.path.exists(index_path):
                index_in_repo = f"{hf_path_prefix.strip('/')}/shard_index.json".lstrip("/")
                HfApi().upload_file(path_or_fileobj=index_path, path_in_repo=index_in_repo, repo_id=hf_repo_id, repo_type="dataset")
                logger.info(f"Uploaded shard_index.json to hf://{hf_repo_id}/{index_in_repo}")

        logger.info(f"Pre-encoding finished: {state['encoded_episodes']} episodes encoded into {state['total_shards_written']} shards")

    def _flush_active_episode(self, state, dataset, output_dir, hf_repo_id, hf_path_prefix, max_shard_bytes, write_executor):
        if state["active_episode_idx"] is None or state["active_buffer"] is None or not state["active_buffer"]["tokens"]:
            return
        buf = state["active_buffer"]
        order = np.argsort(buf["starts"])
        tokens        = np.concatenate([buf["tokens"][i]        for i in order], axis=0)
        actions       = np.concatenate([buf["actions"][i]       for i in order], axis=0)
        states        = np.concatenate([buf["states"][i]        for i in order], axis=0)
        frame_indices = np.concatenate([buf["frame_indices"][i] for i in order], axis=0)
        state["shard_data"].append({
            "episode_idx":   int(state["active_episode_idx"]),
            "sample_idx":    int(dataset.episode_plans[state["active_episode_idx"]]["sample_idx"]),
            "tokens":        tokens,
            "actions":       actions,
            "states":        states,
            "frame_indices": frame_indices,
        })
        state["shard_bytes"] += tokens.nbytes + actions.nbytes + states.nbytes + frame_indices.nbytes
        state["encoded_episodes"] += 1
        if state["shard_bytes"] >= max_shard_bytes:
            if state["pending_write"] is not None:
                state["pending_write"].result()
            state["pending_write"] = write_executor.submit(
                self._write_shard,
                output_dir=output_dir, hf_repo_id=hf_repo_id, hf_path_prefix=hf_path_prefix,
                shard_id=state["shard_id"], shard_data=state["shard_data"],
            )
            state["total_shards_written"] += 1
            state["shard_data"] = []
            state["shard_id"] += 1
            state["shard_bytes"] = 0
        state["active_episode_idx"] = None
        state["active_buffer"] = None

    def _write_shard(self, shard_id, shard_data, output_dir=None, hf_repo_id=None, hf_path_prefix="", hf_repo_type="dataset"):
        shard_name = f"behavior_preencoded_shard_{shard_id:05d}.tar"

        def _save_tar(path):
            """Write each episode as a WebDataset sample inside a tar shard."""
            with tarfile.open(path, "w") as tf:
                for ep in shard_data:
                    key = f"{ep['episode_idx']:06d}"
                    for field in ("tokens", "actions", "states", "frame_indices"):
                        buf = io.BytesIO()
                        np.save(buf, ep[field])
                        data = buf.getvalue()
                        info = tarfile.TarInfo(name=f"{key}.{field}.npy")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
                    meta = json.dumps({"episode_idx": ep["episode_idx"], "sample_idx": ep["sample_idx"]}).encode()
                    info = tarfile.TarInfo(name=f"{key}.json")
                    info.size = len(meta)
                    tf.addfile(info, io.BytesIO(meta))

        episode_index = [{"episode_idx": ep["episode_idx"], "sample_idx": ep["sample_idx"]} for ep in shard_data]
        size_bytes = sum(ep["tokens"].nbytes + ep["actions"].nbytes + ep["states"].nbytes + ep["frame_indices"].nbytes for ep in shard_data)

        if output_dir:
            save_path = os.path.join(output_dir, shard_name)
            _save_tar(save_path)
            self._update_shard_index(output_dir, shard_name, episode_index, size_bytes)
        if hf_repo_id:
            path_in_repo = f"{hf_path_prefix.strip('/')}/{shard_name}".lstrip("/")
            if output_dir:
                upload_path = save_path
            else:
                import tempfile
                tmp_file = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
                tmp_file.close()
                _save_tar(tmp_file.name)
                upload_path = tmp_file.name
            try:
                for attempt in range(3):
                    try:
                        HfApi().upload_file(path_or_fileobj=upload_path, path_in_repo=path_in_repo, repo_id=hf_repo_id, repo_type=hf_repo_type)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        logger.warning(f"HF upload attempt {attempt+1}/3 failed: {e}. Retrying...")
            finally:
                os.remove(upload_path)

    @staticmethod
    def _update_shard_index(output_dir, shard_name, episode_index, size_bytes):
        index_path = os.path.join(output_dir, "shard_index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
        else:
            index = {"shards": []}
        index["shards"].append({
            "file":       shard_name,
            "size_bytes": size_bytes,
            "episodes":   episode_index,
        })
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
