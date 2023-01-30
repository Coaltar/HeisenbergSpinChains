import numpy as np
from common import runtime_decorator

# Constants
spin_x = np.array(([0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]))
spin_y = np.array(([0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]))
spin_z = np.array(([1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]))


def matrix_expansion(matrix_a, matrix_b):
    quadrant_1 = matrix_a[0][0] * matrix_b
    quadrant_2 = matrix_a[0][1] * matrix_b
    quadrant_3 = matrix_a[1][0] * matrix_b
    quadrant_4 = matrix_a[1][1] * matrix_b

    top_half = np.concatenate((quadrant_1, quadrant_2), axis=1)
    bottom_half = np.concatenate((quadrant_3, quadrant_4), axis=1)
    return np.concatenate((top_half, bottom_half))


def elem_from_index(index, i, j, spin):
    if index == i or index == j:
        return spin
    else:
        return np.identity(2)


def axis_inner_product(spin_axis, i, j, particles: int):

    partial_product = elem_from_index(particles - 1, i, j, spin_axis)

    for x in range(particles - 2, -1, -1):
        new_elem = elem_from_index(x, i, j, spin_axis)
        partial_product = matrix_expansion(new_elem, partial_product)

    return partial_product


def hamiltonian_sum(lattice_size, cyclic):
    matrix_size = 2**lattice_size
    matrix_sum = np.zeros((matrix_size, matrix_size))
    for x in range(lattice_size - 1):
        i = x
        j = x + 1
        matrix_sum = np.add(matrix_sum, axis_inner_product(spin_x, i, j, lattice_size))
        matrix_sum = np.add(matrix_sum, axis_inner_product(spin_y, i, j, lattice_size))
        matrix_sum = np.add(matrix_sum, axis_inner_product(spin_z, i, j, lattice_size))
    return matrix_sum


def get_diagonalization(lattice_size: int, cyclic: bool):
    hamiltonian = hamiltonian_sum(lattice_size, cyclic)
    return np.linalg.eig(hamiltonian)


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


if __name__ == "__main__":
    speed_test(10, True)
