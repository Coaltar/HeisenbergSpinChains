# pdm
import lattice_sparse as sp_lat
import matplotlib as mtplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
from common import runtime_decorator
from datatypes import Diagonalized_Hamiltonian, Hamiltonian, LinePlot

# local
import lattice as lat

if __name__ == "__main__":
    print("No main function")


def graph_test(
    graph_title: str,
    x_label: str,
    y_label: str,
    plots: list[LinePlot],
    scale,
    invert_x_axis: bool = False,
):
    data_len = len(plots[0].x_values)

    plot_label_column = []
    x_axis_column = []
    y_axis_column = []

    for plot in plots:
        plot_label_column += [plot.plot_title] * data_len
        x_axis_column += plot.x_values
        y_axis_column += plot.y_values

    d = {
        f"{x_label}": x_axis_column,
        f"{y_label}": y_axis_column,
        "Plot Label": plot_label_column,
    }

    dataframe = pd.DataFrame(data=d)

    sns.set_style("whitegrid")
    plot = sns.lineplot(
        data=dataframe, x=f"{x_label}", y=f"{y_label}", hue="Plot Label", marker="o"
    )
    plot.set(xscale=scale)

    if invert_x_axis:
        plt.invert_xaxis()

    plt.show()
