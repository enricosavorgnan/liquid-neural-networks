import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data import get_data
from src.models import get_model
from src.train import evaluate, train
from src.utils import parse_config, save
from src.vis import plot


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


def main(config: dict):
    seed = config.get("data", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(config.get("device", "cpu"))

    # Simulate data
    train_loader, val_loader, test_loader = get_data(config)

    # Define model
    model = get_model(config).to(device)

    # Train or load model
    train_cfg = config.get("train", {})
    mode = config.get("mode", "train")
    if mode == "train":
        train_performance = train(model=model, train_loader=train_loader, val_loader=val_loader, **train_cfg)
        ckpt_path = Path(train_cfg.get("checkpoint_path", ""))
        if train_cfg.get("checkpoint", False) and ckpt_path.is_file():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif mode == "eval":
        model_path = Path(config.get("save", {}).get("path", ".")) / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        train_performance = None
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'train' or 'eval'.")

    # Evaluate model
    eval_performance = evaluate(
        model=model,
        test_loader=test_loader,
        rollout=train_cfg.get("rollout", True),
        rollout_steps=train_cfg.get("rollout_steps"),
        window=train_cfg.get("window", 64),
    )

    # Plot results
    if config.get("plot", True):
        figs, names = plot(model, train_performance, eval_performance)
    else:
        figs, names = [], []

    # Save model, metrics, and figures
    save(model=model, figs=figs, names=names, performance=[train_performance, eval_performance], config=config)


if __name__ == "__main__":
    config = {
        "mode": "train",
        "device": "xpu",
        "plot": True,
        "data": {
            "name": "lorenz",
            "n_traj": 512,
            "seq_len": 128,
            "dt_min": 0.005,
            "dt_max": 0.06,
            "burn_in": 5.0,
            "split": [0.8, 0.1, 0.1],
            "batch_size": 32,
            "normalize": True,
            "seed": 42,
            "system_kwargs": {},
        },
        "model": {
            "name": "CfC",
            "input_dim": 3,
            "history_dim": 32,
            "output_dim": 3,
            "use_gate": True,
            "backbone_hidden_dim": 64,
            "backbone_output_dim": 32,
            "backbone_activation_func": nn.SiLU,
        },
        "train": {
            "n_epochs": 300,
            "lr": 1e-3,
            "lr_decay": 0.999,
            "rollout": True,
            "rollout_steps": [[1, 120], [2, 60], [4, 60], [8, 30], [16, 30]],
            "window": 64,
            "loss_window_weight": 1.0,
            "val_every": 25,
            "checkpoint": True,
            "checkpoint_path": "exp/cfc-lorenz-no-rollout/best_model.pt",
        },
        "save": {
            "save": True,
            "path": "exp/cfc-lorenz-no-rollout",
        }
    }

    parser = argparse.ArgumentParser(
        description="Closed-form Continuous-Time Neural Networks. Pass --config path/to/config.yaml."
    )
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = parse_config(config)

    main(config)
