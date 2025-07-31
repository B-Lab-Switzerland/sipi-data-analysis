# Std lib imports
from typing import Tuple, Dict, List
from itertools import combinations
from pathlib import Path
from datetime import datetime as dt

# 3rd party imports
import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the values across rows (i.e. within
    any given column) linearly.
    """
    df_interp = df.astype(float).interpolate("cubicspline",axis="rows", limit_area="inside")
    return df_interp


class DataImputer(object):
    """
    Data Imputation via Gaussian Processes.

    This class assumes that the dataset df, for whose columns
    the Gaussian Processes should be fit, has a row index
    of dtype int corresponding to years. In other words, each
    column in df is interpreted as a time series.

    Attributes
    ----------
    df : pandas.DataFrame
        Dataset to be imputed. Notice that a separate
        Gaussian process will be fit for every column
        in df separately. The row index needs to be of
        dtype int and the column index of dtype str.

    measurements : dict[str, List[numpy.ndarray, numpy.ndarray]]
        Dictionary containing column heads as keys and lists 
        as values. These lists contain two elements X and y,
        both numpy.ndarrays, where X are the years at which
        the time series values y are measured.
        
    predictions : dict[str, List[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary containing column heads as keys and lists 
        as values. These lists contain three elements X_common, y_pred,
        and y_std, all numpy.ndarrays, where X_common is the grid of
        values onto which the time series are interpolated, y_pred are
        the interpolated time series values, and y_std the corresponding
        Gaussian process standard deviations (uncertainty envelopes).

    gp_means : pandas.DataFrame
        Pandas.DataFrame of the same structure as df but containing
        the Gaussian Process mean for each column.
        
    gp_std : pandas.DataFrame
        Pandas.DataFrame of the same structure as df but containing
        the Gaussian Process standard deviation (uncertainty envelope)
        for each column.
        
    Methods
    -------
    fit_gp() -> None

    Raises
    ------
    TypeError
        If row index of self.df is not of dtype integer.
    """
    def __init__(self, data: pd.DataFrame):
        if data.index.dtype != int:
            raise TypeError(f"Row index of data is of type {data.index.dtype} but must be 'int'.")
            
        self.df = data
        self.measurements = dict()
        self.predictions = dict()

        # Containers to be filled during method
        # execution
        self.gp_means = pd.DataFrame()
        self.gp_stds = pd.DataFrame()

        # Initialize min_year and max_year
        # These two variables are required to ultimately
        # compute the row index for the resulting output
        # dataframe
        self.min_year = 2025
        self.max_year = 1900

    def _update_year_range(self, X):
        """
        Update the values self.min_year and self.max_year
        containing the minimal and maximal year over all
        columns of self.df.
        """
        if X.min() < self.min_year:
            self.min_year = X.min()
        if X.max() > self.max_year:
            self.max_year = X.max()

    @staticmethod
    def _fit_gp_for_single_column(X_train: np.ndarray,
                                  y_train: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a Gaussian process (GP) to the training
        data given by (X_train, y_train) and predict
        it onto a time grid X_common.

        This methods is ultimately a wrapper around
        GaussianProcessRegressor.fit() and
        GaussianProcessRegressor.predict().

        Parameters
        ----------
        X_train : numpy.ndarray
            Independent variables (features) of the
            training examples.
            
        y_train : numpy.ndarray
            Dependent variable (target) of the
            training examples.

        Returns
        -------
        X_common : numpy.ndarray
            Time grid onto which the GPs are interpolated.
            It contains one entry per month (represented as
            a float corresponding to a fractional year)
            between the first and last year of measurement.
            
        y_pred : numpy.ndarray
            GP mean values predicted for each entry in X_common.
            
        y_std : numpy.ndarray
            GP standard deviation values predicted for each
            entry in X_common.
        """
        # Define a kernel
        kernel = 1.0 * Matern(length_scale=3.3, nu=1.5)
    
        # Fit GP
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-10, normalize_y=True)
        gp.fit(X_train, y_train)
    
        # Predict (grids have different ranges for different metrics, but all have equal spacing)
        X_common = np.arange(X_train.min(), X_train.max() + 1.0/12, 1.0/12).reshape(-1, 1)
        y_pred, y_std = gp.predict(X_common, return_std=True)

        return X_common, y_pred, y_std

    def _setup_result_dfs(self):
        """
        Properly set up the result dataframes self.gp_means
        and self.gp_stds to have the correct column and row
        indices.
        """
        # Create the union of all the different time ranges spanned by
        # the individual columns
        full_range = [np.round(i,2) for i in np.arange(self.min_year,
                                                       self.max_year + 1.0/12,
                                                       1.0/12)
                     ]

        # Now we can use this info to properly set up the result
        # dataframes
        self.gp_means = pd.DataFrame(index=full_range, 
                                     columns=self.df.columns
                                     )

        self.gp_stds = pd.DataFrame(index=full_range, 
                                    columns=self.df.columns
                                   )

    def _fill_interpolations_into_result_dfs(self):
        """
        This function fills the GP-interpolated data
        into the two dataframes self.gp_means and
        self.gp_stds.

        Notice that the GPs were computed on a per-column
        bases. Every column covers a different time range.
        The dataframes self.gp_means and self.gp_stds need
        to be set up before this function is called such that
        their row index corresponds to the union of the time
        ranges of all columns. If that's the case, we can
        assign the GP-interpolation results column-by-column
        and pandas automatically takes care of the putting
        the values into the correct rows.
        """
        if (self.gp_means.shape == (0,0)) or (self.gp_stds.shape == (0,0)):
            raise ValueError("The attributes self.gp_means and/or self.gp_stds"
                             + "have shape (0,0). They need to be setup properly"
                             + "before you can get past this point."
                            )
            
        for metric, gp in self.predictions.items():
            x = [np.round(i,2) for i in gp[0].flatten()]
            y = gp[1]
            e = gp[2]

            # We first create a pandas.Series where each series has its own index
            # This index is a *subset* of self.interp_df.index. This allows us
            # to later just put in this series into the corresponding column of 
            # self.interp_df and pandas automatically will take care of putting
            # the series into the correct rows of the dataframe.
            # REMARK: This approach obviously only works if the index of the Series
            # is actually a subset of the index of the dataframe. To double-check
            # this fact, we put yet another exception handler.
            
            gp_mean = pd.Series(y, index=x, name=metric)
            gp_std = pd.Series(e, index=x, name=metric)

            # checking the rows in gp_mean and self.gp_means are aligned
            if set([i for i in gp_mean.index]).issubset(set([j for j in self.gp_means.index])):
                pass
            else:
                raise ValueError("Indices are not aligned.")
                #raise IndexAlignmentError()

            # If the above test was passed, put the Series into the dataframe.
            self.gp_means[metric] = gp_mean

            # checking the rows in gp_std and self.gp_stds are aligned
            if set([i for i in gp_std.index]).issubset(set([j for j in self.gp_stds.index])):
                pass
            else:
                raise ValueError("Indices are not aligned.")
                #raise IndexAlignmentError()

            # If the above test was passed, put the Series into the dataframe.
            self.gp_stds[metric] = gp_std
        
    def fit_gp(self):
        """
        Fits a separate Gaussian process (GP) for every
        column in the dataset.

        Important: This function only performs interpolations
        and *NOT* extrapolations.
        """
        
        for counter, metric in enumerate(self.df.columns):
            print(f"{counter+1}/{len(self.df.columns)}", end="\r")
            
            # Measurements
            metr_series = self.df[metric].dropna()

            X = np.array([year for year in metr_series.keys()]).reshape(-1, 1)  # GP needs 2D inputs
            y = np.array([val for val in metr_series.values]).ravel()

            # Get year range [min_year, max_year]
            self._update_year_range(X)
            
            # Gather info
            self.measurements[metric] = [X,y]
            self.predictions[metric] = list(self._fit_gp_for_single_column(X,y))

        # Check that key/column names of self.df and self.measurements are aligned
        assert set([metric for metric in self.predictions.keys()]) == set([col for col in self.df.columns])

        # Compile the results into a single dataframe
        self._setup_result_dfs()

        # Fill data column by column into self.gp_means & self.gp_stds 
        self._fill_interpolations_into_result_dfs()
           