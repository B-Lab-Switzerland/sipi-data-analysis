# Std lib imports
from typing import Tuple, Dict, List, Iterable
from collections import namedtuple
from pathlib import Path

# 3rd party imports
import numpy as np
import pandas as pd
from numpy import ndarray
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes


def raw_data_availability_barchart(df: pd.DataFrame, 
                                   x_label: str,
                                   y_label: str,
                                   title: str,
                                   ax: Axes,
                                   show_legend: bool=True) -> Axes:
    """
    Plots availability of raw data in form
    of bar chart. Each metric corresponds
    to a bar and each the length of the bar
    indicates the number of measurements for
    that metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be analyzed

    x_label : str
        Label of plot x axis

    y_label : str
        Label of plot y axis

    title : str
        Title of plot

    ax : Axes
        Axes associated to the canvas the plot
        is drawn on.

    show_legend : bool
        Whether or not a legend is displayed.

    Returns
    -------
    ax : Axes
        Axes containing plot
    """
    # Reset index for plotting
    df_plot = df.reset_index()
    # Use a numeric x axis to avoid categorical spacing issues
    df_plot["y_pos"] = range(len(df_plot))
        
    # Shaded region
    ax.fill_betweenx(y=[0, len(df_plot)], x1=0, x2=10, facecolor="gray", alpha=0.5)
    
    # Plot bars
    sns.barplot(
        data=df_plot,
        x="count", y="y_pos", hue="capital - primary",
        dodge=False,  # Keep all hues at the same x-position
        orient="horizontal",
        ax=ax,
        legend=show_legend
    )
    
    # Shaded region
    ax.axvline(x=10, ls="--", c="k")
    
    # Optional: show fewer x-ticks
    yticks = df_plot["y_pos"]
    ylabels = df_plot["metric_name"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    
    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    
    return ax
    

def visualize_data_availability_colored(df: pd.DataFrame,
                                        x_label: str,
                                        y_label: str,
                                        title: str,
                                        ax: Axes) -> Axes:
    """
    Visualize data availability as a heatmap with custom colors
    for non-null values based on a row attribute (e.g. capital).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to be analyzed

    x_label : str
        Label of plot x axis

    y_label : str
        Label of plot y axis

    title : str
        Title of plot

    ax : Axes
        Axes associated to the canvas the plot
        is drawn on.    

    Returns
    -------
    ax : Axes
        Axes containing plot
    """
    # Map capital categories to colors
    capital_to_color = {'Social': tuple(np.array([177., 41., 48.])/256),
                        'Human': tuple(np.array([40., 96., 144.])/256),
                        'Natural': tuple(np.array([48., 131., 44.])/256),
                        'Economic': tuple(np.array([216., 109., 34.])/256)}

    # Function to apply to each row
    def replace_with_color(row: pd.Series) -> pd.Series:
        """
        Maps data entry to corresponding
        color.

        Parameter
        ---------
        row : pandas.Series
            Single row of the input dataframe containing
            numerical values

        Returns
        -------
        row : pandas.Series
            Single row of the input dataframe with all
            numerical values replaced by a color matching
            the respective capital.
        """
        color = capital_to_color[row["capital"]]
        row = row.apply(lambda val: color if pd.api.types.is_number(val) else val)
        return row.apply(lambda val: (1,1,1) if val=="white" else val)

    colmatr = df.copy()
    colmatr = colmatr.fillna("white")
    
    # Apply to each row
    colmatr = colmatr.apply(replace_with_color, axis=1)
    colmatr = colmatr.drop("capital", axis=1)
    colmatr.columns = pd.to_datetime(colmatr.columns, format="%Y")
    n_rows = len(colmatr.index)
    
    rgb_array = np.array([[tuple(val) for val in row] for row in colmatr.values])

    # Plot the colored "mask"
    years = [dt.year for dt in colmatr.columns]
    xmin = years[0]
    xmax = years[-1] + 1
    ymin = len(colmatr)
    ymax = 0
    ax.imshow(rgb_array, aspect='auto', extent=[xmin, xmax, n_rows - 0.5, -0.5 ])

    # Add grid and ticks
    # Optional: show fewer x-ticks
    ax.grid(True)
    decades = [year for year in years if year % 10 == 0]
    ax.set_xticks(decades)
    ax.set_xticklabels(decades, rotation=45, fontsize=8)

    
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(colmatr.index, fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='') 
               for color in capital_to_color.values()]
    labels = list(capital_to_color.keys())
    ax.legend(handles, labels, title="Capital", loc='lower left', fontsize=8)

    # Save or show
    return ax

def plot_data(line_df: pd.DataFrame, 
              title: str,
              scatter_df: pd.DataFrame|None = None, 
              error_df: pd.DataFrame|None = None,
              fpath: Path|None = None):
    """
    Creates a multi-panel plot showing one time
    series per panel.

    The line_df data is shown as lines, the
    scatter_df data is shown as scatter points
    and error_df is shown as uncertainty envelopes.

    Parameters
    ----------
    line_df : pandas.DataFrame
        Data to be plotted as lines.

    title : str
        Plot title.

    Optional Parameters
    -------------------
    scatter_df : pandas.DataFrame [default: None]
        Data to be plotted as scatter points
        (for reference)

    error_df : pandas.DataFrame [default: None]
        Data to be plotted as uncertainty envelopes
        around line_df.

    fpath : pathlib.Path
        Path to file where plot should be
        saved.

    Raises
    ------
    ValueError
        If error_df is not None and indices of line_df
        and error_df are not aligned.
    """
    fig, axs = plt.subplots(23,5, figsize=(25,60))
    
    i = 0
    for metric in line_df.columns:
        ax = axs[i//5,i%5]

        # Plot lines
        line_series = line_df[metric].dropna()
        line_x = line_series.keys()
        line_y = line_series.values
        ax.plot(line_x, line_y, c="r", label="GP")
        
        if not(scatter_df is None):
            # Plot scatter point data for reference
            scatter_series = scatter_df[metric].dropna()
            scatter_x = scatter_series.keys()
            scatter_y = scatter_series.values
            ax.scatter(scatter_x, scatter_y, c="k", marker='o', label="measurements")

        if not(error_df is None):
            # Add uncertainty envelopes
            error_series = error_df[metric].dropna()
            error_x = error_series.keys()
            error_y = error_series.values

            if not(all(error_x == line_x)):
                raise ValueError("error_df and line_df must have identical indices (x-values).")
            
            ax.fill_between(
                error_x,
                line_y - 1.96 * error_y,
                line_y + 1.96 * error_y,
                alpha=0.3,
                color='blue',
                label="95% confidence interval")
        
        ax.grid(True)
        ax.set_title(metric)
        i += 1

    fig.suptitle(title, fontsize=24, y=1.006)
    plt.tight_layout()
    if not(fpath is None):
        fig.savefig(fpath, bbox_inches="tight")
    plt.show()

    return fig, ax