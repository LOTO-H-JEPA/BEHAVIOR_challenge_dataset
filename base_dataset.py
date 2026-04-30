from .dataset_utils import logger, load_yaml_config


class BaseDataset:
    """
    Base class for dataset handling. Provides common functionality for loading and managing datasets.
    """

    def __init__(self, use_hub_download=False, token=None, **kwargs):
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
        self.dataset_size = dataset_cfg["dataset_size"]
        self.eval_tasks = dataset_cfg["eval_tasks"]
        self.exclude_eval_tasks = dataset_cfg["exclude_eval_tasks"]
        self.include_eval_tasks_fully = dataset_cfg["include_eval_tasks_fully"]
        self.obs_resolution = dataset_cfg["obs_resolution"]
        self.fps = dataset_cfg["fps"]
        self.shard_size = dataset_cfg["shard_size"]
        self.base_dataset_destination = dataset_cfg["base_dataset_destination"]
        self.encoded_dataset_destination = dataset_cfg["encoded_dataset_destination"]
        self.encoded_dataset_destination_path = dataset_cfg[
            "encoded_dataset_destination_path"
        ]
        self.augmentation = dataset_cfg["augmentation"]
        self.encode_dataset = dataset_cfg["encode_dataset"]

        self.repo_id = "behavior-1k/2025-challenge-demos"
        self.use_hub_download = use_hub_download
        self.token = token
        self.logger = logger
        

        # Log a professional summary of the configuration
        self.logger.info("BaseDataset initialized with configuration summary:")
        self.logger.info(f"  camera_view_type: {self.camera_view_type}")
        self.logger.info(f"  dataset_size: {self.dataset_size}")
        self.logger.info(f"  eval_tasks: {self.eval_tasks}")
        self.logger.info(f"  exclude_eval_tasks: {self.exclude_eval_tasks}")
        self.logger.info(f"  include_eval_tasks_fully: {self.include_eval_tasks_fully}")
        self.logger.info(f"  obs_resolution: {self.obs_resolution}")
        self.logger.info(f"  fps: {self.fps}")
        self.logger.info(f"  shard_size: {self.shard_size}")
        self.logger.info(f"  base_dataset_destination: {self.base_dataset_destination}")
        self.logger.info(
            f"  encoded_dataset_destination: {self.encoded_dataset_destination}"
        )
        self.logger.info(
            f"  encoded_dataset_destination_path: {self.encoded_dataset_destination_path}"
        )
        self.logger.info(f"  augmentation: {self.augmentation}")
        self.logger.info(f"  encode_dataset: {self.encode_dataset}")
