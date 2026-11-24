import matplotlib.pyplot as plt
import numpy as np


def plot_state_history(
    states,
    timestamps: np.ndarray,
    horizon: int,
    sample_idx: int = 0,
    max_units: int = 6,
    model_name: str = "Model",
):
    """
    Plot latent hidden-state dynamics for one test trajectory:
      - WC_CfC (dict with 'E' and 'I'): E(t) traces, I(t) traces, and (E_k, I_k) phase portraits.
      - CfC (ndarray (N, L-1, H)): latent unit traces and ||h(t)||_2 norm evolution.
    """
    t = timestamps[sample_idx, 1:]
    L = timestamps.shape[1]
    has_rollout = horizon < L
    t_split = timestamps[sample_idx, horizon] if has_rollout else None

    if isinstance(states, dict) and "E" in states and "I" in states:
        E = np.asarray(states["E"][sample_idx])  # (L - 1, D)
        I = np.asarray(states["I"][sample_idx])  # (L - 1, D)
        n_units = min(max_units, E.shape[1])

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

        for k in range(n_units):
            axes[0].plot(t, E[:, k], lw=1.1, alpha=0.85, label=f"E{k}")
            axes[1].plot(t, I[:, k], lw=1.1, alpha=0.85, label=f"I{k}")
        axes[0].set_title("Excitatory Population E(t)")
        axes[1].set_title("Inhibitory Population I(t)")
        for ax in axes[:2]:
            if t_split is not None:
                ax.axvline(t_split, color="tab:red", ls=":", lw=1.2)
            ax.set_xlabel("t")
            ax.set_ylabel("Activity")
            ax.grid(True, alpha=0.25)

        n_pairs = min(4, E.shape[1])
        for k in range(n_pairs):
            axes[2].plot(E[:, k], I[:, k], lw=1.1, alpha=0.85, label=f"Pair {k}")
            axes[2].scatter(E[0, k], I[0, k], s=18, marker="o")
        axes[2].set_title("E–I Phase Portraits")
        axes[2].set_xlabel("E")
        axes[2].set_ylabel("I")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(fontsize=8)

        fig.suptitle(f"{model_name} — Latent E–I State History (Traj {sample_idx})", y=1.02)
        fig.tight_layout()
        return fig, "state_history"

    # Standard single-population latent state (CfC)
    H = np.asarray(states[sample_idx])  # (L - 1, H_dim)
    n_units = min(max_units, H.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for k in range(n_units):
        axes[0].plot(t, H[:, k], lw=1.1, alpha=0.85, label=f"h{k}")
    axes[0].set_title(f"Latent State Traces (first {n_units} units)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("h(t)")

    h_norm = np.linalg.norm(H, axis=-1)
    axes[1].plot(t, h_norm, color="tab:purple", lw=1.5, label="||h(t)||_2")
    axes[1].set_title("Latent State Norm")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("L2 norm")
    axes[1].legend()

    for ax in axes:
        if t_split is not None:
            ax.axvline(t_split, color="tab:red", ls=":", lw=1.2)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{model_name} — Latent State History (Traj {sample_idx})", y=1.02)
    fig.tight_layout()
    return fig, "state_history"
