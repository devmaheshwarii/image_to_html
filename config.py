import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    # Model parameters
    clip_model_name: str = "ViT-B/32"
    gpt2_model_name: str = "gpt2"
    clip_dim: int = 512
    max_length: int = 512

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 5e-5
    
    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data")
    
    # Device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset_name: str = "HuggingFaceM4/WebSight"
    dataset_version: str = "v0.1"
    dataset_split: str = "train[:0.1%]"

    def __post_init__(self):
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)