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


def tv_problem_2d_chambolle(
    image: np.ndarray, lambda_tv: float = 1.0
) -> Tuple[Callable, Callable, Callable]:
    """
    Create a 2D Total Variation problem instance for image denoising.
    :param image: Noisy input image (grayscale, 2D numpy array)
    Prox calculation using Chambolle's algorithm
    """
    b = image  # Noisy image

    def grad_F(x: np.ndarray) -> np.ndarray:
        """Gradient of the smooth part (data fidelity term)."""
        return x - b

    def prox_J(x: np.ndarray, t: float) -> np.ndarray:
        """Proximal operator for TV norm using Chambolle's algorithm."""
        # Scale the regularization parameter
        tau = t * lambda_tv

        # Initialize the dual variable (p)
        h, w = x.shape
        p = np.zeros((h, w, 2))  # p = (p^1, p^2) dual variable

        # Parameters for Chambolle's algorithm
        max_iter = 30
        tol = 1e-5
        sigma = 1.0 / 8.0  # Step size for dual update (1/8 is stable for 2D)

        # Function to compute divergence of p
        def div(p):
            # Compute the divergence: div(p) = ∂p^1/∂x + ∂p^2/∂y
            div_x = np.zeros_like(x)
            div_y = np.zeros_like(x)

            # Forward differences with Neumann boundary conditions
            div_x[:, :-1] = p[:, :-1, 0] - p[:, 1:, 0]
            div_x[:, -1] = p[:, -1, 0]

            div_y[:-1, :] = p[:-1, :, 1] - p[1:, :, 1]
            div_y[-1, :] = p[-1, :, 1]

            return div_x + div_y

        # Function to compute gradient of x
        def gradient(x):
            # Compute the gradient: ∇x = (∂x/∂x, ∂x/∂y)
            grad = np.zeros((h, w, 2))

            # Backward differences with Neumann boundary conditions
            grad[:, 1:, 0] = x[:, 1:] - x[:, :-1]
            grad[1:, :, 1] = x[1:, :] - x[:-1, :]

            return grad

        # Initialize u with input x
        u = x.copy()

        # Chambolle's algorithm
        for _ in range(max_iter):
            # Compute the gradient of current u
            grad_u = gradient(u)

            # Update the dual variable p
            p_new = p + sigma * grad_u

            # Project p onto the unit ball
            norm = np.sqrt(np.sum(p_new**2, axis=2))
            norm = np.maximum(norm, 1.0)  # Avoid division by zero
            p_new[:, :, 0] /= norm
            p_new[:, :, 1] /= norm

            # Update p
            p = p_new

            # Update the primal variable u
            div_p = div(p)
            u_new = x - tau * div_p

            # Check convergence
            if np.linalg.norm(u_new - u) / np.linalg.norm(u) < tol:
                u = u_new
                break

            u = u_new

        return u

    def obj_phi(x: np.ndarray) -> float:
        """Full objective function: data fidelity + TV norm."""
        # Data fidelity term
        data_term = 0.5 * np.sum((x - b) ** 2)

        # TV norm using finite differences
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(x)

        grad_x[:, 1:] = x[:, 1:] - x[:, :-1]
        grad_y[1:, :] = x[1:, :] - x[:-1, :]

        tv_term = lambda_tv * np.sum(np.sqrt(grad_x**2 + grad_y**2))

        return data_term + tv_term

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
