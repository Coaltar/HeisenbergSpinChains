import logging as LOG
from math import sqrt

import numpy as np
# from numpy import sparse.coo_matrix as COO
import scipy as scp
from datatypes import Hamiltonian
from datetime import datetime


spin_x = scp.sparse.coo_matrix(np.array(([0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j])))
spin_y = scp.sparse.coo_matrix(np.array(([0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j])))
spin_z = scp.sparse.coo_matrix(np.array(([1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j])))

id_2 = scp.sparse.coo_matrix(np.identity(2))


def kronecker_product(matrix_a: scp.sparse.coo_matrix, matrix_b: scp.sparse.coo_matrix):
    """
    This is a type of tensor product.
    """

    # #TODO: Can just replace with numpy kron function
    left_mat_size = matrix_a.shape[0]
    right_mat_size = matrix_b.shape[0]
    new_dimension = left_mat_size * right_mat_size

    new_vals = []
    new_cols = []
    new_rows = []

    for a in range(len(matrix_a.data)):
        for b in range(len(matrix_b.data)):
            new_cols.append(matrix_a.col[a] * right_mat_size + matrix_b.col[b])
            new_rows.append(matrix_a.row[a] * right_mat_size + matrix_b.row[b])
            new_vals.append(matrix_a.data[a] * matrix_b.data[b])

    new_matrix = scp.sparse.coo_matrix(
        (new_vals, (new_rows, new_cols)),
        shape=(new_dimension, new_dimension),
    )

    # print(new_matrix.shape)
    return new_matrix

#


def elem_from_index(index: int, i: int, j: int, spin: scp.sparse.coo_matrix) -> scp.sparse.coo_matrix:
    if index == i or index == j:
        return spin
    else:
        return id_2


def calculate_spin_interaction_matrix(spin_matrix: scp.sparse.coo_matrix, i: int, j: int, particles: int) -> scp.sparse.coo_matrix:
    """
    SiSj contribution to Hamiltonian

    Specific for some i and some j.

    Where Hamiltonian is H = (SiSj)x + (SiSj)y + (SiSj)z
    for some range n, from 0 to L - 1 (except for cyclic case)
    i = n
    j = n + 1

    this is single contribution to the Hamiltonian for a specific spin axis
    for a specific value of i and j
    """

    product = scp.sparse.coo_matrix([[1]])
    for x in range(particles - 1, -1, -1):
        product = kronecker_product(
            elem_from_index(x, i, j, spin_matrix), product)
    return product


def construct_open_hamiltonian(lattice_size: int) -> Hamiltonian:
    """
    Sums the Hamiltonian
    Where Hamiltonian is H = (SiSj)x + (SiSj)y + (SiSj)z
    for some range n, from 0 to L - 1 (except for cyclic case)
    i = n
    j = n + 1
    """
    start = datetime.now()

    matrix_size = 2**lattice_size
    matrix_sum = scp.sparse.coo_matrix(
        ([], ([], [])), shape=(matrix_size, matrix_size))

    for x in range(lattice_size - 1):
        i = x
        j = x + 1
        matrix_sum += calculate_spin_interaction_matrix(
            spin_x, i, j, lattice_size)
        matrix_sum += calculate_spin_interaction_matrix(
            spin_y, i, j, lattice_size)
        matrix_sum += calculate_spin_interaction_matrix(
            spin_z, i, j, lattice_size)
    matrix = scp.sparse.coo_matrix(matrix_sum)
    end = datetime.now()

    time = (end - start).total_seconds()

    return Hamiltonian(size=lattice_size, matrix=matrix, cyclic=False, matrix_diagonal=None, eigenvalues=None,
                       construction_time=time, diagonalization_time=0)


def open_to_closed_hamiltonian(ham: Hamiltonian) -> Hamiltonian:
    """
    Function converts a Hamiltonian for an open chain lattice to it's
    corresponding closed chain counterpart
    """

    new_matrix = np.copy(ham.matrix)
    lattice_size = ham.size

    start = datetime.now()

    x_contrib = calculate_spin_interaction_matrix(
        spin_x, lattice_size - 1, 0, lattice_size)
    y_contrib = calculate_spin_interaction_matrix(
        spin_y, lattice_size - 1, 0, lattice_size)
    z_contrib = calculate_spin_interaction_matrix(
        spin_z, lattice_size - 1, 0, lattice_size)

    new_matrix = np.add(new_matrix, x_contrib)
    new_matrix = np.add(new_matrix, y_contrib)
    new_matrix = np.add(new_matrix, z_contrib)

    end = datetime.now()

    time = ham.construction_time + (end - start).total_seconds()

    return Hamiltonian(size=ham.size, matrix=new_matrix, cyclic=True, matrix_diagonal=None, eigenvalues=None,
                       construction_time=time, diagonalization_time=0)


def diagonalize(ham: Hamiltonian) -> Hamiltonian:
    """
    Takes in a hamiltonian represented as a sparse matrix (coo format)
    Attempts to diagonalize using numpy linalg functions.
    On failure will fall back to normal diagonalization functions.
    """

    # matrix = scp.sparse.coo_matrix(ham.matrix.copy())
    matrix = ham.matrix.copy()

    start = datetime.now()
    try:
        LOG.info("Attempt sparse matrix diagonalization")
        result = scp.sparse.linalg.eigs(matrix)
    except:
        LOG.warning(
            "Sparse matrix diagonalization failed, using regular diagonalization")
        result = np.linalg.eig(matrix.todense())
    finally:
        end = datetime.now()
        eigs, matrix = result

        ham.diagonalization_time = (end - start).total_seconds()
        ham.eigenvalues = eigs
        ham.matrix_diagonal = matrix

        return result


def get_next_hamil(ham: Hamiltonian) -> Hamiltonian:
    """
    Takes in a Hamiltonian H(n) and produces the next
    Hamiltonian H(n+1)
    """
    new_size = ham.size + 1
    old_matrix = scp.sparse.coo_matrix(ham.matrix.copy())

    start = datetime.now()
    new_matrix = kronecker_product(
        old_matrix, id_2) + hamiltonian_partial(new_size)
    end = datetime.now()

    time = ham.construction_time + (end-start).total_seconds()

    return Hamiltonian(size=ham.size+1, matrix=new_matrix, cyclic=False, matrix_diagonal=None, eigenvalues=None,
                       construction_time=time, diagonalization_time=0)


def hamiltonian_partial(size):
    """
    Generates SxSx + SySy + SzSz contribution from the very last indices.
    Used to construct H(n+1) from H(n) for open lattices.
    """
    x_contrib = calculate_spin_interaction_matrix(
        spin_x, size - 2, size - 1, size)
    y_contrib = calculate_spin_interaction_matrix(
        spin_y, size - 2, size - 1, size)
    z_contrib = calculate_spin_interaction_matrix(
        spin_z, size - 2, size - 1, size)

    sum = x_contrib + y_contrib + z_contrib
    return sum


def speed_test(max: int, cyclic: bool):
    print("Re-write test")


if __name__ == "__main__":
    print("no main")
    # speed_test(15, False)
