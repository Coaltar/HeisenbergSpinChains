# import logging as LOG
import logging as LOG
from dataclasses import dataclass
from math import sqrt, pi, log


from pprint import pprint
import lattice_sparse as lat_sparse
import matplotlib as mtplt
import numpy as np
import scipy as scp
import pandas as pd
from common import *
from datatypes import Hamiltonian, DataPlot
from graph import graph_table, multi_graph

import lattice as lat


class LatGen1d:
    def __init__(self, size: int, func_set):
        self.size = size
        self.hamiltonians = []
        self.cyclic_hamiltonians = []
        self.func_set = func_set

    def generate_open_hamiltonians(self):
        """
        Genereates hamiltonians using selected function set
        """

        self.hamiltonians.append(self.func_set.construct_open_hamiltonian(2))
        for n in range(2, self.size):
            index = n - 2
            prev_ham = self.hamiltonians[index]

            next_ham = self.func_set.get_next_hamil(prev_ham)
            self.hamiltonians.append(next_ham)
        for ham in self.hamiltonians:
            self.func_set.diagonalize(ham)

    def generate_cyclic_hamiltonians(self):
        """
        Generates Hamiltonians for closed lattice
        using previously constructed Hamiltonians for open lattice
        """

        for n in range(2, self.size + 1):
            index = n-2
            open_ham = self.hamiltonians[index]
            cylic_ham = self.func_set.open_to_closed_hamiltonian(open_ham)
            self.cyclic_hamiltonians.append(cylic_ham)
        for ham in self.cyclic_hamiltonians:
            self.func_set.diagonalize(ham)
        return
    # ============================
    # Runtime decorated functions
    # Imported from other scripts
    # TODO: There is almost certainly a better way to organize and wrap these functions
    # Maybe a function map instead????
    # ============================

    def show_runtime_graphs(self):

        graph_title = "Runtime Test"
        x_label = "Matrix Dimension"
        y_label = "Construction Runtime"

        open_ham_plot = DataPlot(
            plot_title="Open Lattice",
            x_values=[2**ham.size for ham in self.hamiltonians],
            y_values=[ham.construction_time for ham in self.hamiltonians],
            plot_type="lineplot"
        )

        open_ham_diag_plot = DataPlot(
            plot_title="Diagonalized Open Lattice",
            x_values=[2**ham.size for ham in self.hamiltonians],
            y_values=[ham.diagonalization_time for ham in self.hamiltonians],
            plot_type="lineplot"
        )

        closed_ham_plot = DataPlot(
            plot_title="Closed Lattice",
            x_values=[2**ham.size for ham in self.cyclic_hamiltonians],
            y_values=[ham.construction_time for ham in self.cyclic_hamiltonians],
            plot_type="lineplot"
        )

        closed_ham_diag_plot = DataPlot(
            plot_title="Diagonalized Closed Lattice",
            x_values=[2**ham.size for ham in self.cyclic_hamiltonians],
            y_values=[
                ham.diagonalization_time for ham in self.cyclic_hamiltonians],
            plot_type="lineplot"
        )

        plots = []
        plots.append(open_ham_plot)
        plots.append(open_ham_diag_plot)
        plots.append(closed_ham_plot)
        plots.append(closed_ham_diag_plot)

        base_2_scale = mtplt.scale.LogScale(axis="x", base=2)
        multi_graph(graph_title, x_label, y_label, plots, base_2_scale)
        # graph_lineplots(graph_title, x_label, y_label, plots, base_2_scale)

    @staticmethod
    def energy_relationship(L, E_limit):
        C = 3/8
        # C = 0.3433
        # C = 0.365
        return E_limit - (pi**2/(6*L**2))*(1 + (C / (np.log(L)**3)))

    @staticmethod
    def domain_condition(size):
        if(size % 2 == 0 and size > 2):
            # if(size % 2 == 0):
            return True
        else:
            return False

    def show_curve_fit(self):
        open_chain_energy_per_site_plot = DataPlot(
            plot_title="Open Chain Energy Per Site",
            x_values=[((ham.size))
                      for ham in self.hamiltonians if LatGen1d.domain_condition(ham.size)],
            y_values=[min(
                ham.eigenvalues)/ham.size for ham in self.hamiltonians if LatGen1d.domain_condition(ham.size)],
            plot_type="scatterplot"
        )

        closed_chain_energy_per_site_plot = DataPlot(
            plot_title="Closed Chain Energy Per Site",
            x_values=[((ham.size))
                      for ham in self.cyclic_hamiltonians if LatGen1d.domain_condition(ham.size)],
            y_values=[min(
                ham.eigenvalues)/ham.size for ham in self.cyclic_hamiltonians if LatGen1d.domain_condition(ham.size)],
            plot_type="scatterplot"
        )

        L_vals = []
        y_vals = []

        for ham in self.cyclic_hamiltonians:
            print("???")
            if (LatGen1d.domain_condition(ham.size)):
                L = ham.size
                L_vals.append(L)
                base_e = min(ham.eigenvalues)
                y_vals.append(base_e/L)

        print(L_vals)
        print(y_vals)
        popt, pcov = scp.optimize.curve_fit(
            LatGen1d.energy_relationship, np.array(L_vals), np.array(y_vals))

        energy_limit = popt[0]
        print(energy_limit)

        curve_x = np.linspace(10, 100, 100)
        curve_y = LatGen1d.energy_relationship(curve_x, energy_limit)

        # curve_x = 1 / curve_x**2

        curve_x = curve_x.tolist()
        curve_y = curve_y.tolist()

        # energy_per_site = curve_y

        fit_plot = DataPlot(plot_title="Curve Fit", x_values=curve_x,
                            y_values=curve_y, plot_type="scatterplot")

        plots = []
        graph_title = "Base Energy Per Lattice Site"
        x_label = "1/N^2"
        y_label = "E/N"
        plots.append(open_chain_energy_per_site_plot)
        plots.append(closed_chain_energy_per_site_plot)
        plots.append(fit_plot)

        multi_graph(graph_title, x_label, y_label, plots)

        open_chain_energy_per_site_plot = DataPlot(
            plot_title="Open Chain Energy Per Site",
            x_values=[(1/(ham.size**2))
                      for ham in self.hamiltonians if LatGen1d.domain_condition(ham.size)],
            y_values=[min(
                ham.eigenvalues)/ham.size for ham in self.hamiltonians if LatGen1d.domain_condition(ham.size)],
            plot_type="scatterplot"
        )

        closed_chain_energy_per_site_plot = DataPlot(
            plot_title="Closed Chain Energy Per Site",
            x_values=[(1/(ham.size**2))
                      for ham in self.cyclic_hamiltonians if LatGen1d.domain_condition(ham.size)],
            y_values=[min(
                ham.eigenvalues)/ham.size for ham in self.cyclic_hamiltonians if LatGen1d.domain_condition(ham.size)],
            plot_type="scatterplot"
        )

        L_vals = []
        y_vals = []

        for ham in self.cyclic_hamiltonians:
            if (LatGen1d.domain_condition(ham.size)):
                L = ham.size
                L_vals.append(L)
                base_e = min(ham.eigenvalues)
                y_vals.append(base_e/L)

        popt, pcov = scp.optimize.curve_fit(
            LatGen1d.energy_relationship, np.array(L_vals), np.array(y_vals))

        energy_limit = popt[0]
        print(energy_limit)

        curve_x = np.linspace(10, 100, 100)
        curve_y = LatGen1d.energy_relationship(curve_x, energy_limit)

        curve_x = 1 / curve_x**2

        curve_x = curve_x.tolist()
        curve_y = curve_y.tolist()

        # energy_per_site = curve_y

        fit_plot = DataPlot(plot_title="Curve Fit", x_values=curve_x,
                            y_values=curve_y, plot_type="scatterplot")

        plots = []
        graph_title = "Base Energy Per Lattice Site"
        x_label = "1/N^2"
        y_label = "E/N"
        plots.append(open_chain_energy_per_site_plot)
        plots.append(closed_chain_energy_per_site_plot)
        plots.append(fit_plot)

        multi_graph(graph_title, x_label, y_label, plots)

    def print_vals(self):
        print("Open, noncyclic Hamiltonians")
        print("Lattice Size \t| Base Energy")

        with open("open_chain.txt", "w") as file:
            file.write(("Lattice Size \t| Base Energy\n"))
            for ham in self.hamiltonians:
                size = ham.size
                base = min(ham.eigenvalues)
                base = float(base)
                file.write(f"{size} \t\t, {base}\n")

        for ham in self.hamiltonians:
            size = ham.size
            base = min(ham.eigenvalues)
            base = float(base)
            print(f"{size} \t\t, {base}")

        print("====================")
        print("====================")
        print("====================")

        with open("closed_chain.txt", "w") as file:
            file.write(("Lattice Size \t| Base Energy\n"))
            for ham in self.cyclic_hamiltonians:
                size = ham.size
                base = min(ham.eigenvalues)
                base = float(base)
                file.write(f"{size} \t\t, {base}\n")

        print("Closed, cyclic Hamiltonians")
        print("Lattice Size \t| Base Energy")
        for ham in self.cyclic_hamiltonians:
            size = ham.size
            base = min(ham.eigenvalues)  # todo: remove factor of 4?
            base = float(base)
            print(f"{size} \t\t, {base}")

    def generate_all(self):
        self.generate_open_hamiltonians()
        self.generate_cyclic_hamiltonians()

        # self.diagonalize_hamiltonians()


def demo():
    lattice = LatGen1d(3, lat_sparse)
    lattice.generate_all()

    print(lattice.hamiltonians[1].matrix)
    print(lattice.hamiltonians[1].matrix.todense())
    # lattice.show_runtime_graphs()
    # lattice.show_curve_fit()
    # lattice.print_vals()


if __name__ == "__main__":
    demo()
