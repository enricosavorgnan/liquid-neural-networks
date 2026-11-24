import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def save(model, figs, names, performance, config: dict | None = None, path: str | None = None):
    """Save model weights, performance metrics, and figures (under <exp_dir>/figs/)."""
    save_cfg = config.get("save", {}) if isinstance(config, dict) else {}
    if not save_cfg.get("save", True):
        return

    exp_dir = Path(path or save_cfg.get("path", "."))
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if model is not None:
        torch.save(model.state_dict(), exp_dir / "model.pt")

    for fig, name in zip(figs, names):
        fig.savefig(figs_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    train_perf, eval_perf = performance
    summary = {}
    if isinstance(train_perf, dict):
        summary["train_losses"] = [float(x) for x in train_perf.get("train_losses", [])]
        summary["val_losses"] = [float(x) for x in train_perf.get("val_losses", [])]
        summary["val_epochs"] = [int(x) for x in train_perf.get("val_epochs", [])]
    if isinstance(eval_perf, dict):
        summary["test_loss"] = float(eval_perf["test_loss"])
        summary["horizon"] = int(eval_perf["horizon"])

    with open(exp_dir / "performance.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def resolve_object(name):
    """Resolve a dotted or nn.* symbol string from YAML into a Python object."""
    if not isinstance(name, str):
        return name
    if hasattr(nn, name):
        return getattr(nn, name)
    if "." in name:
        module_name, attr_name = name.rsplit(".", 1)
        if module_name in ("nn", "torch.nn") and hasattr(nn, attr_name):
            return getattr(nn, attr_name)
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    raise ValueError(f"Could not resolve symbol '{name}'")


def parse_config(config: dict) -> dict:
    """Resolve class/activation names loaded as strings from YAML into Python objects."""
    config = dict(config)
    if "model" in config and isinstance(config["model"], dict):
        model_cfg = dict(config["model"])
        if isinstance(model_cfg.get("backbone_activation_func"), str):
            model_cfg["backbone_activation_func"] = resolve_object(model_cfg["backbone_activation_func"])
        config["model"] = model_cfg
    return config
