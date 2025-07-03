# Std lib imports
from typing import Tuple, Dict, List
from itertools import combinations

# 3rd party imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_data_availability(df: pd.DataFrame, 
                                fpath: str=None, 
                                x_label: str=None,
                                y_label: str=None,
                                title: str=None,
                                **kwargs
                               ) -> Tuple:
    """
    Visualize data availability by creating a non-nullity
    matrix (black tiles = non-null values, white tiles = 
    null values).
    """
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.isnull(), 
                cbar=False,        # Hide color bar
                cmap=['black', 'white'],  # black = not null, white = null
                xticklabels = True,
                yticklabels = True,
                ax=ax)
    
    ax.grid(True)

    # Set axis ticks (for x axis: plot only decades)
    tick_locs = [i for i, col in enumerate(df.columns) if col % 10 == 0]
    tick_labels = [str(col) for col in df.columns if col % 10 == 0]
    
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=5)

    # Add axis lables
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add title
    ax.set_title(title)

    # Save plot if desired
    if fpath:
        fig.savefig(fpath)

    plt.show()

    # return
    return fig, ax

def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the values across rows (i.e. within
    any given column) linearly.
    """
    df_interp = df.astype(float).interpolate("linear",axis="rows", limit_area="inside")
    return df_interp

class DataCleaner(object):
    """
    Cleans the dataset passed as df. This class assumes
    that the row index is a vector of integers representing
    years while the column index is a vector of strings.
    """
    def __init__(self, df: pd.DataFrame, verbose: int=1):
        abort = False
        reason = ""
        if ((df.index.dtype!=int) and (df.index.dtype!=np.int64)):
            reason = f"Index has wrong dtype. Is {df.index.dtype} but should be either int or np.int64."
            abort = True
        if df.columns.dtype!='O':
            reason = f"Column has wrong dtype. Is {df.columns.dtype} but should be 'O'."
            abort = True
        if abort:
            raise ValueError("DataFrame has wrong format. Reason:", reason)
             
        self.df = df
        self.verbose = verbose

    def remove_constant_columns(self, threshold: float=1e-8):
        """
        Removes constant columns, i.e. columns that have
        a standard deviation below threshold.
        """
        # Print header message with basic info
        if self.verbose>0:
            header = "Removing constant columns..."
            print(header)
            print("-"*len(header))
            print(f"Before: {len(self.df.columns)} columns")

        # Identifying constant columns
        constant_columns = self.df.loc[:, (self.df.std() <= threshold)].columns

        # Dropping constant columns
        self.df = self.df[[c for c in self.df.columns if c not in constant_columns]].copy()

        # Print more detailed info if desired
        if self.verbose>1:
            if len(constant_columns)>0:
                print(f"Found {len(constant_columns)} constant columns.")
                print("The following columns are constant (std < threshold):")
                for cc in constant_columns:
                    print(f"\t{cc}")
            else:
                print("No constant columns found.")

        # Print footer message with basic info
        if self.verbose>0:
            print(f"After: {len(self.df.columns)} columns")
            footer = "Done.\n"
            print(footer)

    def apply_time_filter(self, min_year: int = 0, max_year: int = 2100):
        """
        Returns only rows with indices >= min_year and
        <= max_year.
        """
        # Print header message with basic info
        if self.verbose>0:
            header = "Removing rows outside defined time windows..."
            print(header)
            print("-"*len(header))
            print(f"Before: {len(self.df.index)} rows")

        # Identifying rows within defined time window
        above_min = self.df.index >= min_year
        below_max = self.df.index <= max_year
        inside_window = above_min & below_max
        

        # Dropping rows outside defined window
        rows2drop = self.df.index[~inside_window]
        self.df = self.df.loc[inside_window, :].copy()

        # Print more detailed info if desired
        if self.verbose>1:
            if sum(~inside_window)>0:
                print(f"Found {sum(~inside_window)} years (i.e. rows) outside of defined time window.")
                print(f"The following years/rows are outside of defined time window [{min_year} - {max_year}]:")
                for row in rows2drop:
                    print(f"\t{row}")
            else:
                print("No rows outside of defined time window found.")

        # Print footer message with basic info
        if self.verbose>0:
            print(f"After: {len(self.df.index)} rows")
            footer = "Done.\n"
            print(footer)

    def drop_sparse_columns(self, n_notnull_min):
        """
        Removes columns that do not have a minimum
        amount of non-null values.
        """
        # Print header message with basic info
        if self.verbose>0:
            header = "Removing sparse columns..."
            print(header)
            print("-"*len(header))
            print(f"Before: {len(self.df.columns)} columns")

        # Identifying sparse columns
        sparse_columns = self.df.loc[:, self.df.count()<3].columns
        
        # Dropping constant columns
        self.df = self.df.drop(sparse_columns, axis=1).copy()

        # Print more detailed info if desired
        if self.verbose>1:
            if len(sparse_columns)>0:
                print(f"Found {len(sparse_columns)} sparse columns.")
                print("The following columns are sparse (#not-null < n_notnull_min):")
                for cc in sparse_columns:
                    print(f"\t{cc}")
            else:
                print("No sparse columns found.")

        # Print footer message with basic info
        if self.verbose>0:
            print(f"After: {len(self.df.columns)} columns")
            footer = "Done.\n"
            print(footer)
            
class CorrelationAnalysis(object):
    """
    """
    def __init__(self, data: pd.DataFrame, name_map=None, timeseries=True):
        self.data = data
        self.is_timeseries = timeseries
        self.name_map = name_map
    
    def compute_correlation(self):
        """
        """
        if self.is_timeseries:
            # use pct_change method to detrend the timeseries
            self.corrmat = self.data.diff().corr()
        else:
            self.corrmat = self.data.corr()

    def plot_corr_heatmap(self, title: str, use_name_map: bool = False, mask: Dict["str", List["str"]] = None):
        """
        """
        df = self.corrmat
        if use_name_map:
            df = df.rename(idx_name_map, axis=1).rename(idx_name_map, axis=0)
            
        fig, ax = plt.subplots(figsize=(10,9))    
        sns.heatmap(df, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)
        plt.tight_layout()

        fig_title = "./" + title
        if self.is_timeseries:
            fig_title += "_detrended"
            
        #fig.savefig(fig_title + ".pdf")

    def drop_strong_correlations(self, threshold: float):
        """
        Filter which indices to keep and which ones to drop
        """
        keepers = dict()
        droppers = dict()
        
        indices = self.corrmat.index
        keepers = []
        droppers = []
        for (i1, i2) in combinations(indices,2):
            keepers.append(i1)
            if np.abs(self.corrmat.loc[i1,i2])>threshold:
                droppers.append(i2)
    
        droppers = set(droppers)
        keepers = list(set(keepers) - droppers)

        return keepers