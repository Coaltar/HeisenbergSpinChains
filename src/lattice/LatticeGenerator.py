# import logging as LOG
import logging as LOG
from dataclasses import dataclass
from math import sqrt

import lattice_sparse as lat_sparse
import matplotlib as mtplt
import numpy as np
import scipy as scp
from common import *
from datatypes import Diagonalized_Hamiltonian, Hamiltonian, LinePlot
from graph import graph_test

import lattice as lat


class LatticeGenerator_1d:
    def __init__(self, size: int, func_set):
        self.size = size

        self.hamiltonians = []
        self.diagonalized_hamiltonians = []

        self.closed_hamiltonians = []
        self.diagonalized_closed_hamiltonians = []

        self.func_set = func_set
        self.spin_x = func_set.spin_x
        self.spin_y = func_set.spin_y
        self.spin_z = func_set.spin_z

    def generate_open_hamiltonians(self):
        """
        Genereates hamiltonians using selected function set
        """

        base_runtime, base_matrix = self.get_base_hamiltonian_matrix()
        base_hamiltonian = Hamiltonian(
            size=2,
            matrix=base_matrix,
            runtime_contribution=base_runtime,
            runtime_cumulative=base_runtime,
        )
        self.hamiltonians.append(base_hamiltonian)

        for n in range(3, self.size + 1):
            previous_index = n - 3
            previous_hamiltonian = self.hamiltonians[previous_index]
            # assert n - 1 == previous_hamiltonian.size

            previous_matrix = previous_hamiltonian.matrix
            runtime_partial, new_matrix = self.get_next_hamil(previous_matrix, n)
            runtime_cumulative = (
                runtime_partial + previous_hamiltonian.runtime_cumulative
            )

            new_hamil = Hamiltonian(
                size=n,
                matrix=new_matrix,
                runtime_contribution=runtime_partial,
                runtime_cumulative=runtime_cumulative,
            )
            self.hamiltonians.append(new_hamil)

    def generate_closed_hamiltonians(self):
        """
        Generates Hamiltonians for closed lattice
        using previously constructed Hamiltonians for open lattice
        """
        assert self.hamiltonians[0] != None

        for hamil in self.hamiltonians:
            matrix = hamil.matrix
            size = hamil.size
            runtime = hamil.runtime_cumulative

            runtime_contribution, new_matrix = self.open_to_closed_hamiltonian(
                matrix, size
            )
            runtime_cumulative = runtime_contribution + runtime

            new_hamil = Hamiltonian(
                size=size,
                matrix=new_matrix,
                runtime_contribution=runtime_contribution,
                runtime_cumulative=runtime_cumulative,
            )
            self.closed_hamiltonians.append(new_hamil)
        return

    def diagonalize_hamiltonians(self):
        for ham in self.hamiltonians:
            matrix = ham.matrix
            runtime_cumulative = ham.runtime_cumulative
            # runtime, res = self.diagonalize_hamiltonian(matrix)
            runtime, (eigs, matrices) = self.diagonalize_hamiltonian(matrix)
            new_diag = Diagonalized_Hamiltonian(
                size=ham.size,
                eigenvalues=eigs,
                matrix=matrices,
                runtime_contribution=runtime,
                runtime_cumulative=runtime_cumulative + runtime,
            )
            self.diagonalized_hamiltonians.append(new_diag)

        for ham in self.closed_hamiltonians:
            matrix = ham.matrix
            runtime_cumulative = ham.runtime_cumulative
            runtime, (eigs, matrices) = self.diagonalize_hamiltonian(matrix)
            new_diag = Diagonalized_Hamiltonian(
                size=ham.size,
                eigenvalues=eigs,
                matrix=matrices,
                runtime_contribution=runtime,
                runtime_cumulative=runtime_cumulative + runtime,
            )
            self.diagonalized_closed_hamiltonians.append(new_diag)

    # ============================
    # Runtime decorated functions
    # Imported from other scripts
    # TODO: There is almost certainly a better way to organize and wrap these functions
    # Maybe a function map instead????
    # ============================

    @runtime_decorator
    def get_base_hamiltonian_matrix(self):
        base_hamiltonian_matrix = self.func_set.open_hamiltonian(2)
        return base_hamiltonian_matrix

    @runtime_decorator
    def get_next_hamil(self, old_matrix: scp.sparse.coo_matrix, new_size: int):
        """
        Generates Hamiltonian H(n+1) from H(n).
        Also wraps it in a runtime decorator.
        """
        new_ham = self.func_set.get_next_hamil(old_matrix, new_size)
        return new_ham

    @runtime_decorator
    def open_to_closed_hamiltonian(
        self, open_ham: scp.sparse.coo_matrix, lattice_size: int
    ):
        open_ham = self.func_set.open_to_closed_hamiltonian(open_ham, lattice_size)
        return open_ham

    @runtime_decorator
    def diagonalize_hamiltonian(self, matrix):
        return self.func_set.diagonalize(matrix)

    def runtime_graphs(self):
        # generate LinePlots
        graph_title = "Runtime Test"
        x_label = "Matrix Dimension"
        y_label = "Construction Runtime"

        plots = []

        open_hams = LatticeGenerator_1d.make_plot(self.hamiltonians, "Open Lattice")
        closed_hams = LatticeGenerator_1d.make_plot(
            self.closed_hamiltonians, "Closed Lattice"
        )
        open_diagonalized_hams = LatticeGenerator_1d.make_plot(
            self.diagonalized_hamiltonians, "Open Lattice, Diagonalized"
        )
        closed_diagonalized_hams = LatticeGenerator_1d.make_plot(
            self.diagonalized_closed_hamiltonians, "Closed Lattice, Diagonalized"
        )
        plots.append(open_hams)
        plots.append(closed_hams)
        plots.append(open_diagonalized_hams)
        plots.append(closed_diagonalized_hams)

        base_2_scale = mtplt.scale.LogScale(axis="x", base=2)
        # plot.set(xticks=elem_count)
        # plot.set(xscale=base_2_scale)
        # g_results.set(xscale='log')
        #
        graph_test(graph_title, x_label, y_label, plots, base_2_scale)

    def energy_graphs(self):
        plots = []

        plot_label = "Open Chain Lattice"
        x_vals = []
        y_vals = []
        for ham in self.diagonalized_hamiltonians:
            # if power_of_2(ham.size):
            if ham.size % 2 == 0 and ham.size != 2:
                base_energy = min(ham.eigenvalues)
                sites = ham.size

                x_vals.append(1 / sqrt(sites))
                y_vals.append(base_energy / sites)
        plot = LinePlot(plot_label, x_vals, y_vals)
        plots.append(plot)

        plot_label = "Closed Chain Lattice"
        x_vals = []
        y_vals = []
        for ham in self.diagonalized_closed_hamiltonians:
            # if power_of_2(ham.size):
            if ham.size % 2 == 0 and ham.size != 2:
                base_energy = min(ham.eigenvalues)
                sites = ham.size

                x_vals.append(1 / sqrt(sites))
                y_vals.append(base_energy / sites)
        plot = LinePlot(plot_label, x_vals, y_vals)
        plots.append(plot)

        graph_title = "Base Energy vs Lattice Size"
        x_label = "1/N"
        y_label = "E/N"
        base_2_scale = mtplt.scale.LogScale(axis="x", base=2)
        graph_test(graph_title, x_label, y_label, plots, base_2_scale)

    @staticmethod
    def make_plot(data_arr: Hamiltonian | Diagonalized_Hamiltonian, plot_label: str):

        x_vals = []
        y_vals = []
        for ham in data_arr:
            size = 2**ham.size
            runtime = ham.runtime_cumulative
            x_vals.append(size)

            y_vals.append(runtime)
        plot = LinePlot(plot_label, x_vals, y_vals)
        return plot

    def print_vals(self):
        for ham in self.diagonalized_hamiltonians:
            size = ham.size
            base = min(ham.eigenvalues)
            print(f"Lattice Size: {size}\nBase State: {base}\n")

        print("====================\n")
        print("====================\n")
        print("====================\n")

        for ham in self.diagonalized_closed_hamiltonians:
            size = ham.size
            base = min(ham.eigenvalues) / 4  # todo: remove factor of 4
            print(f"Lattice Size: {size}\nBase State: {base}\n")

    def generate_all(self):
        self.generate_open_hamiltonians()
        self.generate_closed_hamiltonians()
        self.diagonalize_hamiltonians()


def demo():
    # lattice = LatticeGenerator_1d(7, lat)
    # lattice.generate_all()
    # lattice.runtime_graphs()
    # lattice.print_vals()

    lattice = LatticeGenerator_1d(16, lat_sparse)
    lattice.generate_all()
    # lattice.runtime_graphs()
    lattice.print_vals()
    lattice.energy_graphs()


if __name__ == "__main__":
    demo()
