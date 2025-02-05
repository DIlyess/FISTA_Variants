# problems.py

import numpy as np
from typing import Tuple, Callable
from scipy import linalg


def lasso_problem(
    n: int = 200, sparsity: float = 0.1
) -> Tuple[Callable, Callable, Callable]:
    """
    Create LASSO problem instance
    """
    # Generate synthetic data
    A = np.random.randn(n, n)
    x_true = np.zeros(n)
    k = int(sparsity * n)
    idx = np.random.choice(n, k, replace=False)
    x_true[idx] = np.random.randn(k)
    b = A @ x_true + 0.01 * np.random.randn(n)

    def grad_F(x: np.ndarray) -> np.ndarray:
        """Gradient of the smooth part (least squares)"""
        return A.T @ (A @ x - b)

    def prox_J(x: np.ndarray, t: float) -> np.ndarray:
        """Soft-thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)

    def obj_phi(x: np.ndarray) -> float:
        """Full objective function"""
        return 0.5 * np.sum((A @ x - b) ** 2)

    return grad_F, prox_J, obj_phi


def tv_problem(n: int = 100) -> Tuple[Callable, Callable, Callable]:
    """
    Create Total Variation problem instance
    """
    # Generate synthetic piecewise constant signal
    x_true = np.zeros(n)
    change_points = np.sort(np.random.choice(n, 5, replace=False))
    current_value = 0
    for cp in change_points:
        current_value += np.random.randn()
        x_true[cp:] = current_value
    b = x_true + 0.01 * np.random.randn(n)

    # Finite difference matrix
    D = np.eye(n) - np.eye(n, k=1)[: n - 1]

    def grad_F(x: np.ndarray) -> np.ndarray:
        """Gradient of the smooth part"""
        return x - b

    def prox_J(x: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator for TV norm"""
        z = np.zeros_like(x)
        for i in range(len(x) - 1):
            diff = x[i + 1] - x[i]
            if diff > t:
                z[i + 1] = x[i + 1] - t
                z[i] = x[i]
            elif diff < -t:
                z[i + 1] = x[i + 1] + t
                z[i] = x[i]
            else:
                mean = (x[i + 1] + x[i]) / 2
                z[i + 1] = z[i] = mean
        return z

    def obj_phi(x: np.ndarray) -> float:
        """Full objective function"""
        return 0.5 * np.sum((x - b) ** 2) + np.sum(np.abs(np.diff(x)))

    return grad_F, prox_J, obj_phi


def logistic_regression_problem(
    n_samples: int = 1000, n_features: int = 100
) -> Tuple[Callable, Callable, Callable]:
    """
    Create sparse logistic regression problem instance
    """
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    w_true[np.random.rand(n_features) > 0.1] = 0  # Make it sparse
    p = 1 / (1 + np.exp(-X @ w_true))
    y = (np.random.rand(n_samples) < p).astype(float)

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

    def grad_F(w: np.ndarray) -> np.ndarray:
        """Gradient of the logistic loss"""
        z = X @ w
        return X.T @ (sigmoid(z) - y) / n_samples

    def prox_J(w: np.ndarray, t: float) -> np.ndarray:
        """L1 proximal operator"""
        return np.sign(w) * np.maximum(np.abs(w) - t, 0)

    def obj_phi(w: np.ndarray) -> float:
        """Full objective function"""
        z = X @ w
        log_likelihood = -np.mean(
            y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))
        )
        l1_penalty = np.sum(np.abs(w))
        return log_likelihood + l1_penalty

    return grad_F, prox_J, obj_phi
