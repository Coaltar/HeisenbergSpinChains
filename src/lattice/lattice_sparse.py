import logging as LOG

import numpy as np
import scipy as scp
from common import runtime_decorator

# Constants


spin_x = np.array(([0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]))
spin_y = np.array(([0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]))
spin_z = np.array(([1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]))

spin_x_sparse = scp.sparse.coo_matrix(spin_x)
spin_y_sparse = scp.sparse.coo_matrix(spin_y)
spin_z_sparse = scp.sparse.coo_matrix(spin_z)
sparse_id_2 = scp.sparse.coo_matrix(np.identity(2))


def matrix_expansion(matrix_a, matrix_b):

    quadrant_offset = len(matrix_b.data)

    new_vals = []
    new_cols = []
    new_rows = []

    for a in range(len(matrix_a.data)):
        for b in range(len(matrix_b.data)):

            new_cols.append(matrix_a.col[a] * quadrant_offset + matrix_b.col[b])
            new_rows.append(matrix_a.row[a] * quadrant_offset + matrix_b.row[b])
            new_vals.append(matrix_a.data[a] * matrix_b.data[b])

    new_matrix = scp.sparse.coo_matrix(
        (new_vals, (new_rows, new_cols)),
        shape=(quadrant_offset * 2, quadrant_offset * 2),
    )
    return new_matrix


def elem_from_index(index, i, j, spin):
    if index == i or index == j:
        return spin
    else:
        return sparse_id_2


def axis_inner_product(spin_axis, i, j, particles: int):

    partial_product = elem_from_index(particles - 1, i, j, spin_axis)

    for x in range(particles - 2, -1, -1):
        new_elem = elem_from_index(x, i, j, spin_axis)
        partial_product = matrix_expansion(new_elem, partial_product)

    return partial_product


def hamiltonian_sum(lattice_size, cyclic):
    matrix_size = 2**lattice_size
    # matrix_sum = np.zeros((matrix_size, matrix_size))

    matrix_sum = scp.sparse.coo_matrix(([], ([], [])), shape=(matrix_size, matrix_size))
    for x in range(lattice_size - 1):
        i = x
        j = x + 1
        matrix_sum += axis_inner_product(spin_x_sparse, i, j, lattice_size)
        matrix_sum += axis_inner_product(spin_y_sparse, i, j, lattice_size)
        matrix_sum += axis_inner_product(spin_z_sparse, i, j, lattice_size)
    return matrix_sum


def get_diagonalization(lattice_size: int, cyclic: bool):
    LOG.info(
        f"Getting eigenmatrix and eigenvalues for ${lattice_size} sized particle lattice (cyclic: ${cyclic})"
    )

    hamiltonian = hamiltonian_sum(lattice_size, cyclic)

    # result = scp.sparse.linalg.eigs(hamiltonian)
    # result = np.linalg.eig(hamiltonian.todense())
    # energy = result[0]

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


timed_matrix_expansion = runtime_decorator(matrix_expansion)
timed_hamiltonian_sum = runtime_decorator(hamiltonian_sum)
timed_get_eigenvalues = runtime_decorator(get_diagonalization)


def speed_test(max: int, cyclic: bool):

    for n in range(2, max):

        print("==============================")
        print(f"particles: {n}")
        print("Hamiltonian: ", str(timed_hamiltonian_sum(n, cyclic)[0]))
        print(
            "get_diagonalization: ", str((res := timed_get_eigenvalues(n, cyclic))[0])
        )
        print("energyvalues: ", min(res[1][0]))
        print("\n")

    # print(get_eigenvalues(n, False)[0])


if __name__ == "__main__":
    speed_test(10, True)
