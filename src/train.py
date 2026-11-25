import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm


DEFAULT_ROLLOUT_STEPS = [[1, 200], [2, 100], [4, 50]]


def _concat_state_trajectories(trajectories):
    if isinstance(trajectories[0], dict):
        return {k: torch.cat([t[k] for t in trajectories], dim=1) for k in trajectories[0]}
    return torch.cat(trajectories, dim=1)


def _get_rollout_k(epoch: int, rollout_steps: list[list[int]]) -> int:
    """Return the number of future rollout steps scheduled for the given epoch."""
    elapsed = 0
    for k, duration in rollout_steps:
        elapsed += duration
        if epoch < elapsed:
            return k
    return rollout_steps[-1][0]


def _forward_batch(model: nn.Module,
                   data: torch.Tensor,
                   dt: torch.Tensor,
                   rollout: bool,
                   rollout_steps: list[list[int]],
                   epoch: int,
                   window: int,
                   return_states: bool = False):
    """
    Run window (teacher-forced prefix) + scheduled autoregressive rollout.
    """
    L = data.shape[1]

    # Exactly as rollout = 1: predict only x_{t+1}
    if not rollout or window >= L:
        out = model(data[:, :-1], dt[:, : L - 1], None, return_trajectory=return_states)
        pred_w = out[0]
        target_w = data[:, 1:]
        states = out[2] if return_states else None
        return pred_w, target_w, None, None, pred_w, states

    k = min(_get_rollout_k(epoch, rollout_steps), L - window)

    # Warm-up window
    # We feed the network with real values x_{t} + u_{t-1}
    out = model(data[:, :window], dt[:, :window], None, return_trajectory=return_states)
    prefix, state = out[0], out[1]
    state_trajs = [out[2]] if return_states else None

    pred_w = prefix[:, :-1] if window > 1 else None
    target_w = data[:, 1:window] if window > 1 else None

    # Autoregressive k-steps rollout
    # We feed the network with predicted values \hat{x_{t+1}} and internal states \hat{u_{t}}
    x_current = prefix[:, -1:]
    rollout_preds = [x_current]
    for t in range(window, window + k - 1):
        step = model(x_current, dt[:, t : t + 1], state, return_trajectory=return_states)
        x_current, state = step[0], step[1]
        rollout_preds.append(x_current)
        if return_states:
            state_trajs.append(step[2])

    pred = torch.cat(rollout_preds, dim=1)
    target = data[:, window : window + k]
    full_pred = torch.cat([prefix[:, :-1], pred], dim=1)
    states = _concat_state_trajectories(state_trajs) if return_states else None

    return pred_w, target_w, pred, target, full_pred, states


def _compute_loss(criterion, pred_w, target_w, pred, target, loss_window_weight: float) -> torch.Tensor:
    if pred is None: return criterion(pred_w, target_w)
    loss = criterion(pred, target)
    if pred_w is not None and loss_window_weight > 0.0:
        loss = loss + loss_window_weight * criterion(pred_w, target_w)
    return loss


def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          n_epochs: int = 350,
          lr: float | int = 1e-3,
          lr_decay: float | int = 0.995,
          rollout: bool = True,
          rollout_steps: list[list[int]] | None = None,
          window: int = 64,
          loss_window_weight: float = 1.0,
          val_every: int = 100,
          checkpoint: bool = False,
          checkpoint_path: str = "checkpoint.pt",
          *args, **kwargs):
    # Initial checks
    if rollout_steps is None:
        rollout_steps = [list(stage) for stage in DEFAULT_ROLLOUT_STEPS]
    if rollout:
        total_scheduled = sum(stage[1] for stage in rollout_steps)
        assert total_scheduled == n_epochs, f"Sum of rollout_steps durations ({total_scheduled}) must match n_epochs ({n_epochs})."

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    val_epochs = []
    best_val = float("inf")

    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_losses = []

        for data, dt in train_loader:
            data, dt = data.to(device), dt.to(device)
            pred_w, target_w, pred, target, _, _ = _forward_batch(model, data, dt, rollout=rollout, rollout_steps=rollout_steps, epoch=epoch, window=window)
            loss = _compute_loss(criterion, pred_w, target_w, pred, target, loss_window_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        train_losses.append(float(np.mean(epoch_losses)))

        if epoch % val_every == 0 or epoch == n_epochs - 1:
            model.eval()
            val_loss = []
            # Evaluate validation on the final rollout horizon so val_loss is comparable across all epochs
            val_epoch = n_epochs - 1
            with torch.no_grad():
                for data, dt in val_loader:
                    data, dt = data.to(device), dt.to(device)
                    pred_w, target_w, pred, target, _, _ = _forward_batch(model, data, dt, rollout=rollout, rollout_steps=rollout_steps, epoch=val_epoch, window=window)
                    val_loss.append(_compute_loss(criterion, pred_w, target_w, pred, target, loss_window_weight).item())

            mean_val = float(np.mean(val_loss))
            val_losses.append(mean_val)
            val_epochs.append(epoch)

            if checkpoint and mean_val < best_val:
                best_val = mean_val
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

            k_now = _get_rollout_k(epoch, rollout_steps) if rollout else 1
            print(f"Epoch {epoch}/{n_epochs} (k={k_now}) \t Train Loss: {train_losses[-1]:.4f} \t Val Loss: {mean_val:.4f}")

    return {"train_losses": train_losses, "val_losses": val_losses, "val_epochs": val_epochs}


def _eval_rollouts_single(model: nn.Module,
                          data_1: torch.Tensor,
                          dt_1: torch.Tensor,
                          window: int,
                          ks: list[int]) -> dict[int, np.ndarray]:
    """Evaluate re-anchored k-step autoregressive rollouts over [window .. L-1] on 1 trajectory."""
    L = data_1.shape[1]
    if window >= L:
        return {}

    prefix, init_state = model(data_1[:, :window], dt_1[:, :window], None)
    x_first = prefix[:, -1:]
    n_future = L - window

    rollouts = {}
    for k in ks:
        x_curr = x_first
        state = init_state
        preds = [x_curr]
        for step_idx in range(1, n_future):
            t = window + step_idx - 1
            x_in = data_1[:, t : t + 1] if (step_idx % k == 0) else x_curr
            x_curr, state = model(x_in, dt_1[:, t : t + 1], state)
            preds.append(x_curr)
        rollouts[int(k)] = torch.cat(preds, dim=1)[0].cpu().numpy()

    return rollouts


def evaluate(model: nn.Module,
             test_loader: torch.utils.data.DataLoader,
             rollout: bool = True,
             rollout_steps: list[list[int]] | None = None,
             window: int = 64,
             n_samples: int = 4,
             *args, **kwargs):
    """Evaluate test MSE over the full remaining trajectory after `window` and collect samples for plotting."""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    model.eval()

    ks = [1, 2, 4, 8, 16] if rollout_steps is None else list(dict.fromkeys(int(stage[0]) for stage in rollout_steps))

    test_losses = []
    true_samples = []
    pred_samples = []
    dt_samples = []
    state_samples = []
    rollout_comparison = None
    collected = 0

    with torch.no_grad():
        for data, dt in test_loader:
            data, dt = data.to(device), dt.to(device)
            L = data.shape[1]
            full_rollout_steps = [[max(1, L - window), 1]]
            need_states = collected < n_samples

            pred_w, target_w, pred, target, full_pred, states = _forward_batch(model, data, dt, rollout=rollout, rollout_steps=full_rollout_steps, epoch=0, window=window, return_states=need_states)
            eval_loss = criterion(pred, target) if pred is not None else criterion(pred_w, target_w)
            test_losses.append(eval_loss.item())

            if rollout_comparison is None:
                rollout_comparison = _eval_rollouts_single(model, data[:1], dt[:1], window=window, ks=ks)

            if need_states:
                take = min(n_samples - collected, data.shape[0])
                true_samples.append(data[:take].cpu().numpy())
                pred_samples.append(full_pred[:take].cpu().numpy())
                dt_samples.append(dt[:take].cpu().numpy())
                if isinstance(states, dict):
                    state_samples.append({k: v[:take].cpu().numpy() for k, v in states.items()})
                else:
                    state_samples.append(states[:take].cpu().numpy())
                collected += take

    if isinstance(state_samples[0], dict):
        states_out = {k: np.concatenate([s[k] for s in state_samples], axis=0) for k in state_samples[0]}
    else:
        states_out = np.concatenate(state_samples, axis=0)

    test_mse = float(np.mean(test_losses))
    print(f"Test MSE: {test_mse:.4f}")

    return {
        "test_loss": test_mse,
        "horizon": window if rollout else true_samples[0].shape[1],
        "rollout_window": window,
        "true": np.concatenate(true_samples, axis=0),
        "pred": np.concatenate(pred_samples, axis=0),
        "dt": np.concatenate(dt_samples, axis=0),
        "states": states_out,
        "rollouts": rollout_comparison
    }
