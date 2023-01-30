from datetime import datetime


def runtime_decorator(func):
    def wrap(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        end = datetime.now()

        print(func.__name, delta)

    return wrap


def multiprint(a, b, c):
    print(a)
    print(b)
    print(c)
    return


def compare_matrix_expansion():

    lattice_args = {"matrix_a": lattice.spin_x, "matrix_b": lattice.spin_x}
    runtime_a = measure_runtime(lattice.matrix_expansion, lattice_args)

    lattice_args_sparse = {
        "matrix_a": lattice_sparse.spin_x_sparse,
        "matrix_b": lattice_sparse.spin_x_sparse,
    }
    runtime_b = measure_runtime(lattice_sparse.matrix_expansion, lattice_args)

    print(f"Method a duration: ${runtime_a}")
    print(f"Method b duration: ${runtime_b}")


#
# def compare_hamiltonian_sum():

# def compare_get_eig():


if __name__ == "__main__":
    compare_matrix_expansion()
