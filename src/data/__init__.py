import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.systems import duffing, lorenz, simulate_trajectories


def get_data(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train_loader, val_loader, test_loader) from config['data']."""
    cfg = config["data"]

    states, dts = simulate_trajectories(system=cfg["name"],
                                        n_traj=cfg.get("n_traj", 512),
                                        seq_len=cfg.get("seq_len", 128),
                                        dt_min=cfg.get("dt_min", 0.005),
                                        dt_max=cfg.get("dt_max", 0.06),
                                        burn_in=cfg.get("burn_in", 5.0),
                                        seed=cfg.get("seed", 42),
                                        **cfg.get("system_kwargs", {}) )

    train_frac, val_frac, _ = cfg.get("split", (0.7, 0.15, 0.15))
    n = len(states)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = [(states[:n_train], dts[:n_train]),
              (states[n_train : n_train + n_val], dts[n_train : n_train + n_val]),
              (states[n_train + n_val :], dts[n_train + n_val :])]

    if cfg.get("normalize", True):
        mean = splits[0][0].mean(axis=(0, 1), keepdims=True)
        std = splits[0][0].std(axis=(0, 1), keepdims=True) + 1e-8
        splits = [((s - mean) / std, d) for s, d in splits]

    batch_size = cfg.get("batch_size", 32)
    loaders = [DataLoader(TensorDataset(torch.from_numpy(s), torch.from_numpy(d)),batch_size=batch_size, shuffle=(idx == 0))
        for idx, (s, d) in enumerate(splits)]
    return tuple(loaders)
