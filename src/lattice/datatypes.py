from dataclasses import dataclass

import numpy as np
import scipy as scp


@dataclass
class Hamiltonian:
    size: int
    matrix: scp.sparse.coo_matrix | np.ndarray
    runtime_contribution: float
    runtime_cumulative: float


@dataclass
class Diagonalized_Hamiltonian:
    size: int
    matrix: scp.sparse.coo_matrix | np.ndarray
    eigenvalues: list[float]
    runtime_contribution: float
    runtime_cumulative: float


@dataclass
class LinePlot:
    plot_title: str
    x_values: list[int]
    y_values: list[float]
