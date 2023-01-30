# pdm
import lattice_sparse as sp_lat
import matplotlib as mtplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
from common import runtime_decorator

# local
import lattice as lat

# timed_matrix_expansion = runtime_decorator(lat.matrix_expansion)
timed_ham_sum = runtime_decorator(lat.hamiltonian_sum)
timed_diag = runtime_decorator(lat.get_diagonalization)

sparse_timed_ham_sum = runtime_decorator(sp_lat.hamiltonian_sum)
sparse_timed_diag = runtime_decorator(sp_lat.get_diagonalization)

# print(timed_get_diagonalization(5, False))


def matrix_dimension(matrix):
    # should be using this for the hamiltonian, generally
    dimension = len(matrix**2)
    return dimension


def sparse_matrix_dimension(matrix: scp.sparse.coo_matrix):
    elem_count = len(matrix.data)
    return elem_count


if __name__ == "__main__":

    x_axis = []
    matrix_size = []
    elem_count = []

    ham_speed = []
    diag_speed = []
    ground_states = []

    sparse_ham_speed = []
    sparse_diag_speed = []
    sparse_ground_states = []

    for n in range(2, 11):
        x_axis.append(n)
        matrix_size.append(n**2)

        ham_speed.append(timed_ham_sum(n, False)[0])
        diag_speed.append((diag := timed_diag(n, False))[0])
        ground_states.append(min(diag[1][0]))

        sparse_ham_speed.append((sparse_ham := sparse_timed_ham_sum(n, False))[0])
        sparse_diag_speed.append((sparse_diag := sparse_timed_diag(n, False))[0])
        sparse_ground_states.append(min(diag[1][0]))

        elem_count.append(sparse_matrix_dimension(sparse_ham[1]))

    # print(x_axis)

    # print(ham_speed)
    # print(diag_speed)
    # print(ground_states)

    # print(sparse_ham_speed)
    # print(sparse_diag_speed)
    # print(sparse_ground_states)

    # plt.plot(elem_count, diag_speed, "b", sparse_diag_speed, "g")
    # plt.show()

    ####################
    ## Hamilton Graph
    ###################
    data_len = len(ham_speed)
    type_list = data_len * ["simple"] + data_len * ["sparse"]
    speed_list = ham_speed + sparse_ham_speed
    print(speed_list)
    d = {"elems": 2 * elem_count, "type": type_list, "runtime": speed_list}

    dataframe = pd.DataFrame(data=d)
    d_wide = dataframe.pivot("elems", "type", "runtime")

    # sns.lineplot(data=d_wide)
    sns.set_style("whitegrid")
    plot = sns.lineplot(data=dataframe, x="elems", y="runtime", hue="type", marker="o")

    ## generate base 2 log scale using matplotlib
    # mtplt.scale.LogScale(axis="x", base=2)
    base_2_scale = mtplt.scale.LogScale(axis="x", base=2)
    plot.set(xticks=elem_count)
    plot.set(xscale=base_2_scale)
    # g_results.set(xscale='log')
    plt.show()

    ####################
    ## Diagonlization Graph
    ###################
    data_len = len(diag_speed)
    type_list = data_len * ["simple"] + data_len * ["sparse"]
    speed_list = diag_speed + sparse_diag_speed
    print(speed_list)
    d = {"elems": 2 * elem_count, "type": type_list, "runtime": speed_list}

    dataframe = pd.DataFrame(data=d)
    d_wide = dataframe.pivot("elems", "type", "runtime")

    # sns.lineplot(data=d_wide)
    sns.lineplot(data=dataframe, x="elems", y="runtime", hue="type", marker="o")

    base_2_scale = mtplt.scale.LogScale(axis="x", base=2)
    plot.set(xticks=elem_count)
    plot.set(xscale=base_2_scale)
    # g_results.set(xscale='log')
    plt.show()
