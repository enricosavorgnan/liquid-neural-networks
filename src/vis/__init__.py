import numpy as np

from src.vis.losses import plot_losses
from src.vis.states import plot_state_history
from src.vis.trajectories import plot_rollouts, plot_trajectories


def _build_timestamps(dt: np.ndarray) -> np.ndarray:
    """Convert elapsed intervals dt of shape (N, L - 1, 1) into timestamps (N, L)."""
    dt_2d = dt[..., 0] if dt.ndim == 3 else dt
    zeros = np.zeros((dt_2d.shape[0], 1), dtype=dt_2d.dtype)
    return np.concatenate([zeros, np.cumsum(dt_2d, axis=1)], axis=1)


def _unpack_train_performance(train_performance):
    if isinstance(train_performance, dict):
        train_losses = np.asarray(train_performance["train_losses"], dtype=float)
        val_losses = np.asarray(train_performance["val_losses"], dtype=float)
        val_epochs = np.asarray(train_performance.get("val_epochs", np.arange(len(val_losses))), dtype=int)
    else:
        train_losses, val_losses = train_performance
        train_losses = np.asarray(train_losses, dtype=float)
        val_losses = np.asarray(val_losses, dtype=float)
        if len(val_losses) > 1:
            val_epochs = np.linspace(0, len(train_losses) - 1, len(val_losses), dtype=int)
        else:
            val_epochs = np.zeros(len(val_losses), dtype=int)
    return train_losses, val_losses, val_epochs


def plot(model, train_performance, eval_performance, n_samples: int = 3):
    """Single visualization entry-point: refines metrics/trajectories and delegates to src.vis submodules."""
    figs = []
    names = []
    model_name = type(model).__name__ if model is not None else "Model"

    # 1. Training & validation MSE curves
    if train_performance is not None:
        train_losses, val_losses, val_epochs = _unpack_train_performance(train_performance)
        test_loss = eval_performance.get("test_loss") if isinstance(eval_performance, dict) else None
        fig, name = plot_losses(train_losses, val_losses, val_epochs, test_loss=test_loss, title=f"{model_name} — MSE")
        figs.append(fig)
        names.append(name)

    # 2. Reconstructed trajectories & hidden state history
    if isinstance(eval_performance, dict) and "true" in eval_performance:
        true = np.asarray(eval_performance["true"])
        pred = np.asarray(eval_performance["pred"])
        timestamps = _build_timestamps(np.asarray(eval_performance["dt"]))
        horizon = int(eval_performance.get("horizon", true.shape[1]))

        traj_figs, traj_names = plot_trajectories(
            true=true,
            pred=pred,
            timestamps=timestamps,
            horizon=horizon,
            n_samples=min(n_samples, true.shape[0]),
            model_name=model_name,
        )
        figs.extend(traj_figs)
        names.extend(traj_names)

        if eval_performance.get("rollouts"):
            roll_figs, roll_names = plot_rollouts(
                true_traj=true[0],
                rollouts=eval_performance["rollouts"],
                timestamps=timestamps[0],
                window=int(eval_performance.get("rollout_window", horizon)),
                model_name=model_name,
            )
            figs.extend(roll_figs)
            names.extend(roll_names)

        if eval_performance.get("states") is not None:
            fig, name = plot_state_history(
                states=eval_performance["states"],
                timestamps=timestamps,
                horizon=horizon,
                model_name=model_name,
            )
            figs.append(fig)
            names.append(name)

    return figs, names
