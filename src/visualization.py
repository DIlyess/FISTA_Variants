import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def plot_convergence(
    histories: List[Dict], labels: List[str], title: str = "Convergence Analysis"
):
    """Plot convergence behavior of different FISTA variants"""
    plt.figure(figsize=(12, 8))

    # Plot objective values
    plt.subplot(2, 1, 1)
    for history, label in zip(histories, labels):
        objectives = np.array(history["objective"])
        min_obj = min(objectives)
        plt.semilogy(objectives - min_obj, label=label)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Iterations")
    plt.ylabel("$\Phi(x_k)$")
    plt.legend()
    plt.title(f"{title} - Objective Value")

    # Plot residuals
    plt.subplot(2, 1, 2)
    for history, label in zip(histories, labels):
        plt.semilogy(history["residual"], label=label)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Iterations")
    plt.ylabel("$\|x_k - x_{k-1}\|_2$")
    plt.legend()
    plt.title(f"{title} - Residual")

    plt.tight_layout()
    plt.show()
