# optimizer.py

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class OptimizationParams:
    """Parameters for optimization algorithms"""

    dim: int  # dimension of the problem
    mu: float  # regularization parameter
    gamma: float  # step size
    tol: float = 1e-15  # tolerance for stopping criterion
    max_iter: int = 50000  # maximum number of iterations
    verbose: bool = True
    x0: Optional[np.ndarray] = None  # initial point


class BaseFISTA(ABC):
    """Base class for FISTA variants"""

    def __init__(self, params: OptimizationParams):
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """Validate optimization parameters"""
        if self.params.gamma <= 0:
            raise ValueError("Step size gamma must be positive")
        if self.params.mu < 0:
            raise ValueError("Regularization parameter mu must be non-negative")
        if self.params.tol <= 0:
            raise ValueError("Tolerance must be positive")

    def _initialize(self) -> np.ndarray:
        """Initialize starting point"""
        if self.params.x0 is None:
            return np.zeros(self.params.dim)
        return self.params.x0.copy()

    @abstractmethod
    def optimize(
        self, grad_F: Callable, prox_J: Callable, obj_phi: Callable
    ) -> Tuple[np.ndarray, dict]:
        """Main optimization method to be implemented by subclasses"""
        pass


class FistaBT(BaseFISTA):
    """Original FISTA with backtracking"""

    def optimize(
        self, grad_F: Callable, prox_J: Callable, obj_phi: Callable
    ) -> Tuple[np.ndarray, dict]:
        # Initialize parameters
        x = self._initialize()
        y = x.copy()
        t = 1.0

        # History for tracking convergence
        history = {"objective": [], "residual": [], "iterations": 0}

        for k in range(self.params.max_iter):
            # Store old values
            x_old = x.copy()

            # Forward-backward step
            grad = grad_F(y)
            x = prox_J(y - self.params.gamma * grad, self.params.mu * self.params.gamma)

            # Update momentum term
            t_old = t
            t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            y = x + ((t_old - 1) / t) * (x - x_old)

            # Calculate residual
            residual = np.linalg.norm(x - x_old)

            # Track progress
            if self.params.verbose and k % 100 == 0:
                print(f"Iteration {k}: residual = {residual:.3e}")

            history["objective"].append(obj_phi(x))
            history["residual"].append(residual)

            # Check convergence
            if residual < self.params.tol:
                break

        history["iterations"] = k + 1
        return x, history


class FistaMod(BaseFISTA):
    """Lazy-start FISTA variant"""

    def __init__(
        self,
        params: OptimizationParams,
        p: float = 1 / 20,
        q: float = 1 / 2,
        r: float = 4,
    ):
        super().__init__(params)
        self.p = p
        self.q = q
        self.r = r

    def optimize(
        self, grad_F: Callable, prox_J: Callable, obj_phi: Callable
    ) -> Tuple[np.ndarray, dict]:
        x = self._initialize()
        y = x.copy()
        t = 1.0

        history = {"objective": [], "residual": [], "iterations": 0}

        for k in range(self.params.max_iter):
            x_old = x.copy()

            # Forward-backward step
            grad = grad_F(y)
            x = prox_J(y - self.params.gamma * grad, self.params.mu * self.params.gamma)

            # Modified momentum term
            t_old = t
            t = (self.p + np.sqrt(self.q + self.r * t_old**2)) / 2
            a = (t_old - 1) / t  ################### faut il mettre un min avec 1 ?
            y = x + a * (x - x_old)

            residual = np.linalg.norm(x - x_old)

            if self.params.verbose and k % 100 == 0:
                print(f"Iteration {k}: residual = {residual:.3e}")

            history["objective"].append(obj_phi(x))
            history["residual"].append(residual)

            if residual < self.params.tol:
                break

        history["iterations"] = k + 1
        return x, history


class RestartingFISTA(BaseFISTA):
    """FISTA with adaptive restart"""

    def optimize(
        self, grad_F: Callable, prox_J: Callable, obj_phi: Callable
    ) -> Tuple[np.ndarray, dict]:
        x = self._initialize()
        y = x.copy()
        t = 1.0

        history = {"objective": [], "residual": [], "iterations": 0}

        for k in range(self.params.max_iter):
            x_old = x.copy()
            y_old = y.copy()

            # Forward-backward step
            grad = grad_F(y)
            x = prox_J(y - self.params.gamma * grad, self.params.mu * self.params.gamma)

            # Check restart condition
            if np.dot(y_old - x, x - x_old) > 0:
                t = 1.0
                y = x.copy()
            else:
                t_old = t
                t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
                y = x + ((t_old - 1) / t) * (x - x_old)

            residual = np.linalg.norm(x - x_old)

            if self.params.verbose and k % 100 == 0:
                print(f"Iteration {k}: residual = {residual:.3e}")

            history["objective"].append(obj_phi(x))
            history["residual"].append(residual)

            if residual < self.params.tol:
                break

        history["iterations"] = k + 1
        return x, history


class GreedyFISTA(BaseFISTA):
    """Greedy FISTA variant"""

    def __init__(self, params: OptimizationParams, c_gamma: float = 1.3):
        super().__init__(params)
        self.c_gamma = c_gamma

    def optimize(
        self, grad_F: Callable, prox_J: Callable, obj_phi: Callable
    ) -> Tuple[np.ndarray, dict]:
        x = self._initialize()
        y = x.copy()
        gamma = self.params.gamma * self.c_gamma

        history = {"objective": [], "residual": [], "iterations": 0}

        for k in range(self.params.max_iter):
            x_old = x.copy()
            y_old = y.copy()

            # Forward-backward step with adaptive step size
            grad = grad_F(y)
            x = prox_J(y - gamma * grad, self.params.mu * gamma)

            # Adaptive momentum
            a = 1.0  # Can be modified based on iteration number
            y = x + a * (x - x_old)

            # Step size adaptation
            if np.linalg.norm(x - x_old) > np.linalg.norm(y_old - x_old):
                gamma = max(self.params.gamma, gamma * 0.96)
                y = x.copy()

            residual = np.linalg.norm(x - x_old)

            if self.params.verbose and k % 100 == 0:
                print(f"Iteration {k}: residual = {residual:.3e}")

            history["objective"].append(obj_phi(x))
            history["residual"].append(residual)

            if residual < self.params.tol:
                break

        history["iterations"] = k + 1
        return x, history
