# Std lib imports
from typing import Tuple, Dict, List
from itertools import combinations
from pathlib import Path

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
    df_interp = df.astype(float).interpolate("cubicspline",axis="rows", limit_area="inside")
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

    def drop_duplicates(self) -> pd.DataFrame:
        """
        Drops duplicate rows.
        """
        if self.verbose>0:
            header = "Removing duplicated rows..."
            print(header)
            print("-"*len(header))
            print(f"Before: {len(self.df)} rows")

        duplicates = self.df.loc[self.df.duplicated(keep=False),:].sort_index()
        n_duplicated = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()

        if self.verbose>1:
            if n_duplicated>0:
                print(f"Found {n_duplicated} duplicated rows.")
                print("The following rows are duplicated:")
                display(duplicates)
            else:
                print("No duplicated rows found.")

        # Print footer message with basic info
        if self.verbose>0:
            print(f"After: {len(self.df)} rows")
            footer = "Done.\n"
            print(footer)

        return duplicates

    def remove_constant_columns(self, threshold: float=1e-8) -> List[str]:
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

        return [c for c in constant_columns]
        
    def apply_time_filter(self, min_year: int = 0, max_year: int = 2100) -> List[str]:
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

        return [r for r in rows2drop]

    def drop_sparse_columns(self, n_notnull_min) -> List[str]:
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
        sparse_columns = self.df.loc[:, self.df.count()<n_notnull_min].columns
        
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

        return [c for c in sparse_columns]
            
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

    def plot_corr_heatmap(self, title: str, use_name_map: bool = False, mask: Dict["str", List["str"]] = None, fpath: Path=None):
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
            fig_title += " (detrended)"

        if fpath:
            fig.savefig(fpath)

    def _sort_corrmat_by_variances(self, ascending: bool=False) -> pd.DataFrame:
        """
        """
        # Compute variances of values for each metric
        varsvec = self.data.var()
        
        # Next create a matrix with the variances of each metric in the first column
        # and the covariances/correlations in all the subsequent columns.
        # Sort the resulting matrix by variance in descending order.
        # This step is performed to sort the rows of the correlation matrix by
        # the variances of the respective metrics.
        varcovar = varsvec.to_frame()\
                          .merge(self.corrmat, left_index=True, right_index=True)\
                          .rename({0: "variance"}, axis=1)\
                          .sort_values(by="variance", ascending=ascending)
        
        # Now that the correlation matrix is sorted, we can drop the variance column again
        varsorted = varcovar.drop("variance", axis=1)

        return varsorted 

    def drop_strong_correlations(self, threshold: float, verbose: int=0, fpath_corr=None):
        """
        Filter which indices to keep and which ones to drop
        """

        # Setup
        varsorted_corrmat = self._sort_corrmat_by_variances(ascending=False)
        n_start = len(varsorted_corrmat)        
        n_remaining = n_start
        n_removed = 0
        to_drop = []

        if verbose>0:
            print(f"Starting with {n_remaining} metrics.")

        i = 0
        correlation_xlsx = dict()
        while i < len(varsorted_corrmat):
            # Get current test metric name
            curr_metric = varsorted_corrmat.index[i]
            
            # Look at the current row of the variance-sorted
            # correlation matrix
            curr_corrs = varsorted_corrmat.iloc[i,:]
        
            # In this row, extract those correlations whose absolute
            # value exceeds the threshold
            candidates = curr_corrs[curr_corrs.abs()>threshold]
        
            # As the correlation of a metric with itself is always
            # 1, the current test metric itself will be in the list 
            # of candidates. Obviously, the current test metric 
            # should not be dropped. Therefore, we need to exclude it.
            to_drop_ser = candidates[candidates.keys()!=curr_metric]
        
            # OPTIONAL: round correlation values to two decimals for
            # better readability
            to_drop_ser = to_drop_ser.apply(lambda x: round(x,2))

            if verbose>0:
                # Print all metrics that should be dropped because they
                # are correlated more strongly than threshold (in absolute
                # value-terms) with the current test metric and are therefore
                # considered redundant.
                print(f"{to_drop_ser.name} (var = {monet_vars.sort_values(ascending=False).iloc[i].round(3)}): {to_drop_ser.to_dict()}")

            if len(to_drop_ser)>0:
                correlation_xlsx[to_drop_ser.name] = to_drop_ser
            
            # It follows a sanity check. If any of the metrics to be
            # dropped is already in the to_drop list, this means we've
            # dropped it before. As a result, we really shouldn't be 
            # looking at it again. The fact that we are doing just that
            # means that something went wrong before (namely, we did not
            # drop it after all). In this case, break the loop as there
            # is a logic error that needs fixing.
            if any([d in to_drop for d in to_drop_ser]):
                if verbose>0:
                    print(to_drop_ser)
                    print(to_drop)
                raise ValueError("Trying to drop a metric that should no longer be considered.")
        
            # Add the new set of redundant metrics to the list "to_drop"
            # and update the number of metrics to be removed
            to_drop.extend(list(to_drop_ser.keys()))
            n_removed += len(to_drop_ser)
            if verbose>0:
                print(f"Removing {n_removed} metrics... {n_remaining}-{n_removed}={n_remaining-n_removed}")
        
            # Also, keep a tally of metrics we want to keep
            to_keep = [m for m in varsorted_corrmat.index if not(m in to_drop)]
            n_remaining -= len(to_drop_ser)
        
            # Now we need to actually drop the metrics, i.e. remove them
            # from the correlation matrix before the next iteration. As the 
            # correlation matrix is quadratic, the metrics need to be dropped
            # both from the rows and the columns.
            varsorted_corrmat = varsorted_corrmat.loc[to_keep, to_keep] 

            if verbose>0:
                print(f"Remaining: {len(varcovar)} metrics")
                print()

            # Update index
            i += 1

        if fpath_corr:
            with pd.ExcelWriter(fpath_corr) as writer:
                for sheet_name, ser in correlation_xlsx.items():
                    ser.to_frame().to_excel(writer, sheet_name=sheet_name, index=True)

        # Sanity checks
        # =============
        kept_metrics = [c for c in varsorted_corrmat.columns]
        if kept_metrics != to_keep:
            raise ValueError("kept_metrics != to_keep")
        if set(to_keep).intersection(set(to_drop)) != set():
            raise ValueError("set(to_keep).intersection(set(to_drop)) != set()")
        if len(to_keep)+len(to_drop) != n_start:
            raise ValueError("len(to_keep)+len(to_drop) != n_start")

        return to_keep, correlation_xlsx