import logging as LOG
from math import sqrt

import numpy as np
import scipy as scp
from common import runtime_decorator

# Constants


spin_x = scp.sparse.coo_matrix(np.array(([0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j])))
spin_y = scp.sparse.coo_matrix(np.array(([0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j])))
spin_z = scp.sparse.coo_matrix(np.array(([1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j])))

id_2 = scp.sparse.coo_matrix(np.identity(2))


def matrix_expansion(matrix_a: scp.sparse.coo_matrix, matrix_b: scp.sparse.coo_matrix):
    """
    This is a type of tensor product.
    todo: Specify????
    """

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
    return new_matrix


def elem_from_index(index, i, j, spin):
    if index == i or index == j:
        return spin
    else:
        return id_2


def spin_operator_product(spin_operator, i, j, particles: int):
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

    partial_product = elem_from_index(particles - 1, i, j, spin_operator)

    for x in range(particles - 2, -1, -1):
        new_elem = elem_from_index(x, i, j, spin_operator)
        partial_product = matrix_expansion(new_elem, partial_product)

    return partial_product


def open_hamiltonian(lattice_size: int) -> scp.sparse.coo_matrix:
    """
    Sums the Hamiltonian
    Where Hamiltonian is H = (SiSj)x + (SiSj)y + (SiSj)z
    for some range n, from 0 to L - 1 (except for cyclic case)
    i = n
    j = n + 1
    """

    matrix_size = 2**lattice_size
    matrix_sum = scp.sparse.coo_matrix(([], ([], [])), shape=(matrix_size, matrix_size))
    for x in range(lattice_size - 1):
        i = x
        j = x + 1
        matrix_sum += spin_operator_product(spin_x, i, j, lattice_size)
        matrix_sum += spin_operator_product(spin_y, i, j, lattice_size)
        matrix_sum += spin_operator_product(spin_z, i, j, lattice_size)
    return scp.sparse.coo_matrix(matrix_sum)


def open_to_closed_hamiltonian(open_ham: scp.sparse.coo_matrix, lattice_size: int):
    """
    Function converts a Hamiltonian for an open chain lattice to it's corresponding closed chain counterpart/
    """

    x_contrib = spin_operator_product(spin_x, lattice_size - 1, 0, lattice_size)
    y_contrib = spin_operator_product(spin_y, lattice_size - 1, 0, lattice_size)
    z_contrib = spin_operator_product(spin_z, lattice_size - 1, 0, lattice_size)
    open_ham = np.add(open_ham, x_contrib)
    open_ham = np.add(open_ham, y_contrib)
    open_ham = np.add(open_ham, z_contrib)
    return open_ham


def diagonalize(hamiltonian: scp.sparse.coo_matrix):
    """
    Takes in a hamiltonian represented as a sparse matrix (coo format)
    Attempts to diagonalize using numpy linalg functions.
    On failure will fall back to normal diagonalization functions.
    """

    try:
        LOG.info("Attempt sparse matrix diagonalization")
        result = scp.sparse.linalg.eigs(hamiltonian)
    except:
        LOG.warning(
            "Sparse matrix diagonalization failed, using regular diagonalization"
        )
        result = np.linalg.eig(hamiltonian.todense())
    finally:
        return result


def get_next_hamil(old_matrix: scp.sparse.coo_matrix, new_size: int):
    """
    Takes in a Hamiltonian H(n) and produces the next
    Hamiltonian H(n+1)
    """
    new_ham = matrix_expansion(old_matrix, id_2) + hamiltonian_partial(new_size)
    return scp.sparse.coo_matrix(new_ham)  # todo, this conversion might be expensive


def hamiltonian_partial(size):
    """
    Generates SxSx + SySy + SzSz contribution from the very last indices.
    Used to construct H(n+1) from H(n)
    """
    x_contrib = spin_operator_product(spin_x, size - 2, size - 1, size)
    y_contrib = spin_operator_product(spin_y, size - 2, size - 1, size)
    z_contrib = spin_operator_product(spin_z, size - 2, size - 1, size)

    sum = x_contrib + y_contrib + z_contrib
    return sum


def speed_test(max: int, cyclic: bool):

    LOG.warning("BROKEN, re-write")
    for n in range(2, max):

        print("==============================")
        print(f"particles: {n}")
        print("Hamiltonian: ", str(timed_open_hamiltonian(n)[0]))
        print(
            "get_diagonalization: ", str((res := timed_get_eigenvalues(n, cyclic))[0])
        )
        print("energyvalues: ", min(res[1][0]))
        print("\n")

    # print(get_eigenvalues(n, False)[0])


if __name__ == "__main__":
    print("no main")
    # speed_test(15, False)
