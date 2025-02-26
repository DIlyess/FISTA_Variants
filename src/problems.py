# problems.py

import numpy as np
from typing import Tuple, Callable
from scipy import linalg
from scipy.ndimage import convolve


def lasso_problem(
    dim: int, n: int = 200, sparsity: float = 0.1
) -> Tuple[Callable, Callable, Callable]:
    """
    Create LASSO problem instance
    """
    # Generate synthetic data
    A = np.random.randn(n, dim)
    x_true = np.zeros(dim)
    k = int(sparsity * dim)
    idx = np.random.choice(dim, k, replace=False)
    x_true[idx] = np.random.randn(k)
    b = A @ x_true + 10 * np.random.randn(n)

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


def tv_problem_2d(image: np.ndarray) -> Tuple[Callable, Callable, Callable]:
    """
    Create a 2D Total Variation problem instance for image denoising.
    :param image: Noisy input image (grayscale, 2D numpy array)
    """
    b = image  # Noisy image
    m, n = b.shape  # Image dimensions

    # Finite difference operators for horizontal and vertical gradients
    Dx = np.array([[1, -1]])  # Horizontal difference
    Dy = np.array([[1], [-1]])  # Vertical difference

    def grad_F(x: np.ndarray) -> np.ndarray:
        """Gradient of the smooth part (data fidelity term)."""
        return x - b

    def prox_J(x: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator for TV norm using anisotropic total variation."""
        grad_x = convolve(x, Dx, mode="nearest")  # Horizontal gradient
        grad_y = convolve(x, Dy, mode="nearest")  # Vertical gradient

        # Soft-thresholding
        grad_x = np.sign(grad_x) * np.maximum(np.abs(grad_x) - t, 0)
        grad_y = np.sign(grad_y) * np.maximum(np.abs(grad_y) - t, 0)

        # Compute divergence (negative adjoint of gradient)
        div_x = convolve(grad_x, -Dx, mode="nearest")
        div_y = convolve(grad_y, -Dy, mode="nearest")

        return x + t * (div_x + div_y)

    def obj_phi(x: np.ndarray) -> float:
        """Full objective function: data fidelity + TV norm."""
        tv_term = np.sum(np.abs(convolve(x, Dx, mode="nearest"))) + np.sum(
            np.abs(convolve(x, Dy, mode="nearest"))
        )
        return 0.5 * np.sum((x - b) ** 2) + tv_term

    return grad_F, prox_J, obj_phi


def logistic_regression_problem(
    dim: int, n_samples: int = 1000, n_features: int = 100
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
