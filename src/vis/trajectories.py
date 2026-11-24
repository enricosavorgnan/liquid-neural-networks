import matplotlib.pyplot as plt
import numpy as np


def _coord_labels(dim: int) -> list[str]:
    if dim == 2:
        return ["x", "v"]
    if dim == 3:
        return ["x", "y", "z"]
    return [f"dim {d}" for d in range(dim)]


def plot_trajectories(
    true: np.ndarray,
    pred: np.ndarray,
    timestamps: np.ndarray,
    horizon: int,
    n_samples: int = 3,
    model_name: str = "Model",
):
    """
    Plot reconstructed vs. ground-truth trajectories:
      1. Per-coordinate time series against irregular timestamps.
      2. 2D or 3D phase-space portraits.
    """
    n_samples = min(n_samples, true.shape[0])
    L, dim = true.shape[1], true.shape[2]
    labels = _coord_labels(dim)
    has_rollout = horizon < L

    # 1. Per-coordinate time series
    fig_ts, axes = plt.subplots(n_samples, dim, figsize=(4.2 * dim, 2.6 * n_samples), squeeze=False)
    for i in range(n_samples):
        t = timestamps[i]
        for d in range(dim):
            ax = axes[i, d]
            ax.plot(t, true[i, :, d], color="black", linewidth=1.5, label="True")
            ax.plot(t[1:], pred[i, :, d], color="tab:blue", linestyle="--", linewidth=1.3, label="Pred")
            if has_rollout:
                ax.axvline(t[horizon], color="tab:red", linestyle=":", linewidth=1.2, label="Forecast start")
            if i == 0:
                ax.set_title(labels[d])
            if d == 0:
                ax.set_ylabel(f"Traj {i}")
            if i == n_samples - 1:
                ax.set_xlabel("t")
            ax.grid(True, alpha=0.25)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig_ts.legend(handles, leg_labels, loc="upper right", ncol=3, frameon=False)
    fig_ts.suptitle(f"{model_name} — Reconstructed Trajectories", y=1.01)
    fig_ts.tight_layout()

    # 2. Phase-space portraits (2D or 3D)
    is_3d = dim >= 3
    fig_phase = plt.figure(figsize=(4.5 * n_samples, 4.2))
    for i in range(n_samples):
        ax = fig_phase.add_subplot(1, n_samples, i + 1, projection="3d" if is_3d else None)
        if is_3d:
            ax.plot(true[i, :, 0], true[i, :, 1], true[i, :, 2], color="black", lw=1.2, alpha=0.8, label="True")
            ax.plot(pred[i, :, 0], pred[i, :, 1], pred[i, :, 2], color="tab:blue", ls="--", lw=1.2, label="Pred")
            if has_rollout:
                ax.scatter(*true[i, horizon, :3], color="tab:red", s=25, label="Forecast start")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
        else:
            ax.plot(true[i, :, 0], true[i, :, 1], color="black", lw=1.4, alpha=0.85, label="True")
            ax.plot(pred[i, :, 0], pred[i, :, 1], color="tab:blue", ls="--", lw=1.3, label="Pred")
            if has_rollout:
                ax.scatter(true[i, horizon, 0], true[i, horizon, 1], color="tab:red", s=30, zorder=5, label="Forecast start")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.grid(True, alpha=0.25)
        ax.set_title(f"Traj {i}")
        if i == 0:
            ax.legend(fontsize=8)

    fig_phase.suptitle(f"{model_name} — Phase Portraits", y=1.01)
    fig_phase.tight_layout()

    return [fig_ts, fig_phase], ["trajectories_timeseries", "trajectories_phase"]


def plot_rollouts(
    true_traj: np.ndarray,
    rollouts: dict[int, np.ndarray],
    timestamps: np.ndarray,
    window: int,
    model_name: str = "Model",
):
    """
    Compare re-anchored k-step rollouts (k = 1, 2, 4, 8, 16, ...) on 1 single trajectory:
      1. Per-dimension time series (rows = k values, cols = dimensions).
      2. Phase-space portraits (1 row of 2D/3D subplots across k values).
    """
    ks = list(rollouts.keys())
    n_k = len(ks)
    true_eval = true_traj[window:]
    t_eval = timestamps[window:]
    dim = true_eval.shape[1]
    labels = _coord_labels(dim)

    # 1. Per-dimension time series across rollout horizons k
    fig_ts, axes = plt.subplots(n_k, dim, figsize=(4.2 * dim, 2.2 * n_k), squeeze=False)
    for row, k in enumerate(ks):
        pred_k = rollouts[k]
        mse_k = float(np.mean((pred_k - true_eval) ** 2))
        for d in range(dim):
            ax = axes[row, d]
            ax.plot(t_eval, true_eval[:, d], color="black", lw=1.4, label="True")
            ax.plot(t_eval, pred_k[:, d], color="tab:blue", ls="--", lw=1.2, label="Pred")
            if row == 0:
                ax.set_title(labels[d])
            if d == 0:
                ax.set_ylabel(f"k={k}\nMSE={mse_k:.3g}")
            if row == n_k - 1:
                ax.set_xlabel("t")
            ax.grid(True, alpha=0.25)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig_ts.legend(handles, leg_labels, loc="upper right", ncol=2, frameon=False)
    fig_ts.suptitle(f"{model_name} — Rollout Comparison (Time Series)", y=1.01)
    fig_ts.tight_layout()

    # 2. Phase-space portraits across rollout horizons k
    is_3d = dim >= 3
    fig_phase = plt.figure(figsize=(3.8 * n_k, 3.8))
    for idx, k in enumerate(ks):
        pred_k = rollouts[k]
        mse_k = float(np.mean((pred_k - true_eval) ** 2))
        ax = fig_phase.add_subplot(1, n_k, idx + 1, projection="3d" if is_3d else None)
        if is_3d:
            ax.plot(true_eval[:, 0], true_eval[:, 1], true_eval[:, 2], color="black", lw=1.2, alpha=0.8, label="True")
            ax.plot(pred_k[:, 0], pred_k[:, 1], pred_k[:, 2], color="tab:blue", ls="--", lw=1.2, label="Pred")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
        else:
            ax.plot(true_eval[:, 0], true_eval[:, 1], color="black", lw=1.3, alpha=0.85, label="True")
            ax.plot(pred_k[:, 0], pred_k[:, 1], color="tab:blue", ls="--", lw=1.2, label="Pred")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.grid(True, alpha=0.25)
        ax.set_title(f"k={k} (MSE={mse_k:.3g})")
        if idx == 0:
            ax.legend(fontsize=8)

    fig_phase.suptitle(f"{model_name} — Rollout Comparison (Phase Space)", y=1.02)
    fig_phase.tight_layout()

    return [fig_ts, fig_phase], ["rollouts_timeseries", "rollouts_phase"]
