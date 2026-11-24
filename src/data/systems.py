import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def duffing(t, state, delta=0.2, alpha=1.0, beta=1.0, gamma=0.0, omega=1.2):
    x, v = state
    return [v, -delta * v - alpha * x - beta * (x ** 3) + gamma * np.cos(omega * t)]


SYSTEMS = {
    "lorenz": (lorenz, 3),
    "duffing": (duffing, 2)
}


def simulate_trajectories(system: str,
                          n_traj: int = 512,
                          seq_len: int = 128,
                          dt_min: float = 0.005,
                          dt_max: float = 0.06,
                          burn_in: float = 5.0,
                          seed: int = 42,
                          **system_kwargs
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate irregularly sampled trajectories for a dynamical system.

    Returns:
        states: (n_traj, seq_len, dim) float32 array at timestamps t_0 .. t_{L-1}
        dts:    (n_traj, seq_len - 1, 1) float32 array of elapsed times dt_k = t_{k+1} - t_k
    """
    name = system.lower()
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system '{system}'. Expected one of {list(SYSTEMS)}.")

    rhs_fn, dim = SYSTEMS[name]
    rhs = (lambda t, y: rhs_fn(t, y, **system_kwargs)) if system_kwargs else rhs_fn

    rng = np.random.default_rng(seed)
    states = np.empty((n_traj, seq_len, dim), dtype=np.float32)
    dts = np.empty((n_traj, seq_len - 1, 1), dtype=np.float32)

    for i in range(n_traj):
        if name == "lorenz":
            y0 = rng.normal(0.0, 5.0, size=3) + np.array([0.0, 0.0, 25.0])
        else:
            y0 = rng.normal(0.0, 1.0, size=dim)

        dt_seq = rng.uniform(dt_min, dt_max, size=seq_len - 1)
        t_eval = burn_in + np.concatenate(([0.0], np.cumsum(dt_seq)))

        sol = solve_ivp(rhs, t_span=(0.0, float(t_eval[-1])), y0=y0, t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed on trajectory {i}: {sol.message}")

        states[i] = sol.y.T
        dts[i, :, 0] = dt_seq

    return states, dts


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    states, deltas = simulate_trajectories(system="lorenz", n_traj=5, seq_len=128, burn_in=5.0, dt_min=0.005, dt_max=0.06)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d" if states.shape[-1] == 3 else None)
    for traj in states:
        ax.plot(*traj.T)
    plt.show()
