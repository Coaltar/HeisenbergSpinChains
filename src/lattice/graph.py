# pdm
import lattice_sparse as sp_lat
import matplotlib as mtplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
from common import runtime_decorator
from datatypes import Hamiltonian, DataPlot

# local
import lattice as lat

if __name__ == "__main__":
    print("No main function")


def plot_set(plots: list[DataPlot], plot_type: str) -> None:

    graphing_map = {
        "lineplot": sns.lineplot,
        "scatterplot": sns.scatterplot
    }

    plot_label_column = []
    plot_style_column = []
    x_axis_column = []
    y_axis_column = []
    for plot in plots:
        data_len = len(plot.x_values)
        plot_label_column += [plot.plot_title] * data_len
        # plot_style_column += [plot.style] * data_len
        x_axis_column += plot.x_values
        y_axis_column += plot.y_values

    d = {
        "x": x_axis_column,
        "y": y_axis_column,
        "Plot Label": plot_label_column,
        # "Style": plot_style_column
    }

    dataframe = pd.DataFrame(data=d)

    # markers_dict = {
    #     "solid": 'o',
    #     "dots": 'o',
    #     "dashed": 'o'
    # }

    # sns.set_style("whitegrid")
    graphing_map[plot_type](data=dataframe, x="x", y="y", hue="Plot Label")


def multi_graph(
        graph_title: str,
        x_label: str,
        y_label: str,
        plots: list[DataPlot],
        scale="linear",
        invert_x_axis: bool = False) -> None:

    scatterplots = [plot for plot in plots if plot.plot_type == "scatterplot"]
    lineplots = [plot for plot in plots if plot.plot_type == "lineplot"]

    plot_set(scatterplots, "scatterplot")
    plot_set(lineplots, "lineplot")
    sns.set_style("whitegrid")
    plt.show()
    # lineplot.show()
    #
    #


def graph_table(data, columns):

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    df = pd.DataFrame(data, columns=columns)
    ax.table(cellText=data.astype(float).values,
             colLabels=df.columns, loc='center')

    plt.show()
