from dataclasses import dataclass

import numpy as np
import scipy as scp


@dataclass
class Hamiltonian:
    size: int
    cyclic: bool
    matrix: scp.sparse.coo_matrix | np.ndarray
    matrix_diagonal: scp.sparse.coo_matrix | np.ndarray | None
    eigenvalues: np.ndarray | None
    construction_time: float
    diagonalization_time: float


# @dataclass
# class Diagonalized_Hamiltonian:
#     size: int
#     matrix: scp.sparse.coo_matrix | np.ndarray
#     eigenvalues: list[float]
#     runtime_contribution: float
#     runtime_cumulative: float


@dataclass
class DataPlot:
    plot_title: str
    x_values: list[int] | list[float]
    y_values: list[float]
    plot_type: str
