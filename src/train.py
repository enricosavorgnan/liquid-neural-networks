import torch

from src.models.cfc import *
from src.models.wccfc import *

def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          n_epochs: int,
          lr: float | int,
          lr_decay: float | int,
          checkpoint : bool,
          checkpoint_path : str,
          *args, **kwargs):
    pass
