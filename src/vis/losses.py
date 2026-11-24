import matplotlib.pyplot as plt
import numpy as np


def plot_losses(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_epochs: np.ndarray,
    test_loss: float | None = None,
    title: str = "Training & Validation MSE",
):
    fig, ax = plt.subplots(figsize=(7, 4))

    epochs = np.arange(len(train_losses))
    ax.semilogy(epochs, train_losses, label="Train MSE", linewidth=1.6)
    if len(val_losses) > 0:
        ax.semilogy(val_epochs, val_losses, "o-", label="Val MSE", linewidth=1.6, markersize=4)
    if test_loss is not None:
        ax.axhline(test_loss, linestyle="--", color="tab:red", linewidth=1.2, label=f"Test MSE ({test_loss:.4g})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, "losses"
