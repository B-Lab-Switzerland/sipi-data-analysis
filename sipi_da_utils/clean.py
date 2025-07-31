# Std lib imports
from typing import Tuple, Dict, List
from itertools import combinations
from pathlib import Path
from datetime import datetime as dt

# 3rd party imports
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa import seasonal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    
class DataCleaner(object):
    """
    Cleans the dataset passed as df. This class assumes
    that the row index is a vector of integers representing
    years while the column index is a vector of strings.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be cleaned

    verbose : int
        Level of standard output verbosity during code
        execution.
        
    Methods
    -------
    drop_duplicates() -> pd.DataFrame
    remove_constant_columns(threshold: float=1e-8) -> List[str]
    apply_time_filter(min_year: int = 0, max_year: int = 2100) -> List[str]
    drop_sparse_columns(n_notnull_min) -> List[str]
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
        Drops duplicate rows and returns the rows
        that are droped (i.e. this function behaves
        similar to a pop in this respect).

        Returns
        -------
        duplicates : pandas.DataFrame
            DataFrame containing all the rows that
            originally were duplicates and are now
            being droped.
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

        Optional Parameters
        -------------------
        threshold : float [default: 1e-8]
            Data columns with a standard deviation smaller or equal
            to the threshold are considered being constant as their
            variance is not considered to be significant enough.

        Returns
        -------
        constant_columns : Listen [str]
            List of constant colum headers. 
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

        Optional Parameters
        -------------------
        min_year : int [default = 0]
            Data rows with index smaller than min_year
            will be dropped.
            
        max_year : int [default = 2100]
            Data rows with index greater than max_year
            will be dropped.

        Returns
        -------
        rows2drop : List[str]
            List of row names that were outside of the
            defined time window [min_year, max_year].
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
        rows2drop = [r for r in self.df.index[~inside_window]]
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

        return rows2drop

    def drop_sparse_columns(self, n_notnull_min: int) -> List[str]:
        """
        Removes columns that are too sparse, i.e. columns
        that do not have a minimum amount of non-null values.

        Parameters
        ----------
        n_notnull_min : int
            Minimally required number of non-null values for
            a given column to be considered not sparse.

        Returns
        -------
        sparse_columns : List[str]
            List of sparse column headers.
        """
        # Print header message with basic info
        if self.verbose>0:
            header = "Removing sparse columns..."
            print(header)
            print("-"*len(header))
            print(f"Before: {len(self.df.columns)} columns")

        # Identifying sparse columns
        sparse_columns = [c for c in self.df.loc[:, self.df.count()<n_notnull_min].columns]
        
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

        return sparse_columns