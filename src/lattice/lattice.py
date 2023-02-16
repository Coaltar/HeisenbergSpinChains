import numpy as np
from common import runtime_decorator

# Constants
spin_x = np.array(([0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]))
spin_y = np.array(([0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]))
spin_z = np.array(([1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]))

id_2 = np.identity(2)

"""
The functions here are mostly unoptimized.
This file is kept for the sake of runtime comparisons with lattice_sparse.py
"""


def matrix_expansion(matrix_a, matrix_b):
    """
    Performs a type of tensor product on two matrices.
    Brute force tensor product using nested loop.
    Very bad.
    """

    row_elems = []
    for row in matrix_a:
        col_elems = []
        for col in row:
            col_elems.append(col * matrix_b)
        row_elems.append(np.concatenate(col_elems, axis=1))
    final = np.concatenate(row_elems)

    return final


def elem_from_index(index, i, j, spin):
    """
    Returns a spin matrix or the 2x2 identity matrix.
    Could probably be a one-line conditional.
    """
    if index == i or index == j:
        return spin
    else:
        return np.identity(2)


def spin_operator_product(spin_operator, i, j, particles: int):
    """
    Gets the contribution of a given spin operator at a particular site (represented by i & j)
    ex, for Sx1,Sx2, on a lattice of size 4
    Would be a tensor product of:
        I * Sx * Sx * I
    Where I is the 2x2 Identity matrix.
    """

    partial_product = elem_from_index(particles - 1, i, j, spin_operator)

    for x in range(particles - 2, -1, -1):
        new_elem = elem_from_index(x, i, j, spin_operator)
        partial_product = matrix_expansion(new_elem, partial_product)

    return partial_product


def open_hamiltonian(lattice_size):
    """
    Constructs the hamiltonian for an open lattice given lattice_size
    """
    matrix_size = 2**lattice_size
    matrix_sum = np.zeros((matrix_size, matrix_size))
    for x in range(lattice_size - 1):
        i = x
        j = x + 1
        matrix_sum = np.add(
            matrix_sum, spin_operator_product(spin_x, i, j, lattice_size)
        )
        matrix_sum = np.add(
            matrix_sum, spin_operator_product(spin_y, i, j, lattice_size)
        )
        matrix_sum = np.add(
            matrix_sum, spin_operator_product(spin_z, i, j, lattice_size)
        )
    return matrix_sum


def open_to_closed_hamiltonian(open_ham, lattice_size):
    """
    Generates the Hamiltonian for cyclic/closed lattice
    using the Hamiltonian for an open lattice.
    """

    x_contrib = spin_operator_product(spin_x, lattice_size - 1, 0, lattice_size)
    y_contrib = spin_operator_product(spin_y, lattice_size - 1, 0, lattice_size)
    z_contrib = spin_operator_product(spin_z, lattice_size - 1, 0, lattice_size)

    open_ham = np.add(open_ham, x_contrib)
    open_ham = np.add(open_ham, y_contrib)
    open_ham = np.add(open_ham, z_contrib)
    return open_ham


def diagonalize(hamiltonian):
    """
    Takes in a hamiltonian represented as a matrix
    diagonalizes and returns result
    """

    return np.linalg.eig(hamiltonian)


def get_diagonalization(lattice_size: int):
    hamiltonian = open_hamiltonian(lattice_size)
    return np.linalg.eig(hamiltonian)


def get_next_hamil(old_matrix: np.ndarray, new_size: int):
    """
    Taking the previous Hamiltonian, constructs the next Hamiltonian in the sequence.
    """

    prev_con = matrix_expansion(old_matrix, id_2)
    new_con = hamiltonian_partial(new_size)
    new_ham = prev_con + new_con
    return new_ham


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


if __name__ == "__main__":
    print("No main")
    # speed_test(10, True)
