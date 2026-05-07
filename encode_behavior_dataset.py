#!/usr/bin/env python3

import argparse

import torch
import yaml
from transformers import AutoModel, AutoVideoProcessor

from behavior import BehaviorEpisodePreencoder, BehaviorVideoDataset


class HFVJEPA2Encoder(torch.nn.Module):
    """Wrapper exposing HF V-JEPA2 encoder features as a plain tensor."""

    def __init__(self, hf_repo_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_repo_id)
        # Loaded for parity with official usage and future pre-processing hooks.
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo_id)

    def forward(self, video):
        outputs = self.model(pixel_values=video)
        return outputs.last_hidden_state


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    meta_cfg = cfg.get("meta", {})
    out_cfg = cfg["output"]

    dtype_name = meta_cfg.get("dtype", "float32").lower()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_repo_id = model_cfg.get("hf_repo", "facebook/vjepa2-vitg-fpc64-256")
    encoder = HFVJEPA2Encoder(hf_repo_id=hf_repo_id).to(device)

    # Keep transform optional in this script; callers can pass preprocessed videos if needed.
    transform = None
    dataset = BehaviorVideoDataset(
        data_path=data_cfg["datasets"][0],
        fpcs=data_cfg["dataset_fpcs"][0],
        fps=data_cfg.get("fps"),
        transform=transform,
        camera_frame=data_cfg.get("camera_frame", False),
        state_start_idx=data_cfg.get("state_start_idx", 0),
        state_dim=data_cfg.get("state_dim", 7),
        action_dim=data_cfg.get("action_dim", 23),
    )

    preencoder = BehaviorEpisodePreencoder(encoder=encoder, device=device, dtype=dtype)
    preencoder.encode_full_episodes(
        dataset,
        output_dir=out_cfg.get("local_output_dir"),
        hf_repo_id=out_cfg.get("hf_repo_id"),
        hf_path_prefix=out_cfg.get("hf_path_prefix", ""),
        episodes_per_shard=out_cfg.get("episodes_per_shard", 1),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_mem", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="YAML config path")
    args = parser.parse_args()
    main(args.fname)
