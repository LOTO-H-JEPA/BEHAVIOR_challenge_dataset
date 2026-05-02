# BEHAVIOR_challange_dataset

## Why you were seeing `data/data`, `meta/meta`, `video/videos` and `task-XXXX` folders

The downloaded paths come from the source dataset templates in metadata (`data_path`, `metainfo_path`, `video_path`). Those remote paths already include nested segments such as `data/task-0033/...` and `videos/task-0033/...`. When we were passing a local destination folder (e.g., `<task>/data`) directly into `hf_hub_download`, Hugging Face preserved the remote path under that destination, which produced structures like:

- `<task>/data/data/task-0033/...`
- `<task>/meta/meta/episodes/task-0033/...`
- `<task>/video/videos/task-0033/...`

The downloader now flattens each file into the intended local folder (`data`, `meta`, `video`) by copying only the downloaded filename into that folder.

## Run the pipeline in Google Colab

Use this in a Colab notebook:

```bash
# 1) Clone repo
!git clone <YOUR_REPO_URL>
%cd BEHAVIOR_challenge_dataset

# 2) Install deps
!pip install -r requirements.txt

# 3) (Optional) set HF token for private/gated repos
import os
os.environ["HF_TOKEN"] = "<your_token>"

# 4) Run pipeline
!python -m main
```

### Optional: edit dataset config first

Open `configs/dataset.yaml` and set:

- `base_dataset_destination`
- `dataset_size`
- `camera_view_type`
- `exclude_eval_tasks`

Then rerun:

```bash
!python -m main
```
