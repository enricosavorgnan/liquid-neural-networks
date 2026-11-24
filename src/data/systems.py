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
