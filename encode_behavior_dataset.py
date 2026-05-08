#!/usr/bin/env python3

import argparse
import os
import torch
import yaml
from transformers import AutoModel, AutoVideoProcessor
from vjepa2_BEHAVIOR.app.vjepa_droid.transforms import make_transforms

from behavior import BehaviorEpisodePreencoder, BehaviorVideoDataset


class HFVJEPA2Encoder(torch.nn.Module):
    """Wrapper exposing HF V-JEPA2 encoder features as a plain tensor."""

    def __init__(self, hf_repo_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_repo_id)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo_id)

    def forward(self, video):
      #print("[HFVJEPA2] raw input:", tuple(video.shape), flush=True)
      if video.ndim != 5:
          raise ValueError(f"Expected 5D video tensor, got {tuple(video.shape)}")
      # behavior.py currently gives us [B, C, T, H, W].
      # Hugging Face VJEPA2 expects [B, T, C, H, W].
      if video.shape[1] == 3:
          video = video.permute(0, 2, 1, 3, 4).contiguous()
      elif video.shape[2] == 3:
          video = video.contiguous()
      else:
          raise ValueError(f"Unable to infer channel axis, got {tuple(video.shape)}")

      print("[HFVJEPA2] for HF:", tuple(video.shape), flush=True)

      outputs = self.model(pixel_values_videos=video)
      return outputs.last_hidden_state


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    meta_cfg = cfg.get("meta", {})
    out_cfg = cfg["output"]
    aug_cfg = cfg.get("data_aug", {})

    dtype_name = meta_cfg.get("dtype", "float32").lower()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_repo_id = model_cfg.get("hf_repo", "facebook/vjepa2-vitg-fpc64-256")
    encoder = HFVJEPA2Encoder(hf_repo_id=hf_repo_id).to(device)
    
    transform = make_transforms(
        random_horizontal_flip=aug_cfg.get("horizontal_flip", False),
        random_resize_aspect_ratio=tuple(aug_cfg.get("random_resize_aspect_ratio", [0.75, 1.35])),
        random_resize_scale=tuple(aug_cfg.get("random_resize_scale", [1.777, 1.777])),
        reprob=aug_cfg.get("reprob", 0.0),
        auto_augment=aug_cfg.get("auto_augment", False),
        motion_shift=aug_cfg.get("motion_shift", False),
        crop_size=data_cfg["crop_size"],
    )
    dataset = BehaviorVideoDataset(
        data_path=data_cfg["datasets"][0],
        fpcs=data_cfg["dataset_fpcs"][0],
        fps=data_cfg.get("fps"),
        transform=transform,
        camera_frame=data_cfg.get("camera_frame", False),
        state_start_idx=data_cfg.get("state_start_idx", 0),
        state_dim=data_cfg.get("state_dim", 7),
        action_dim=data_cfg.get("action_dim", 23),
        cache_parquet=data_cfg.get("cache_parquet", False),
        cache_video_readers=data_cfg.get("cache_video_readers", False),
    )

    preencoder = BehaviorEpisodePreencoder(encoder=encoder, device=device, dtype=dtype)
    max_workers = max(1, ((os.cpu_count() or 1) - 1))
    configured_workers = data_cfg.get("num_workers", 4)
    num_workers = min(configured_workers, max_workers)

    preencoder.encode_full_episodes(
        dataset,
        output_dir=out_cfg.get("local_output_dir"),
        hf_repo_id=out_cfg.get("hf_repo_id"),
        hf_path_prefix=out_cfg.get("hf_path_prefix", ""),
        episodes_per_shard=out_cfg.get("episodes_per_shard", 1),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_mem", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="YAML config path")
    args = parser.parse_args()
    main(args.fname)
