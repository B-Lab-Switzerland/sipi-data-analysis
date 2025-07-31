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

def fractional_years_to_datetime(fractional_years: pd.Series) -> pd.Series:
    """
    Convert fractional years to proper datetime format.

    Converts a series of fractional years into a proper datatime
    format consisting of the year and the months (the day is 
    always set to 1, i.e. the first day of the month).

    Example: 1988.833 --> 1988-10-01

    Parameters
    ----------
    fractional_years : pandas.Series
        Series containing all the fractional years to be
        converted.

    Returns
    -------
    dts : pandas.Series
        Series containing the correct datetime-formatted
        years/months.
    """
    int_years = fractional_years.astype(int)
    months = (np.round((fractional_years - int_years)*12)+1).astype(int)
    dts = pd.to_datetime({'year': int_years, 'month': months, 'day': 1 })
    return dts

class TSAnalyzer(object):
    """
    Time Series Analyzer.

    
    """
    def __init__(self, data: pd.DataFrame, dirpath: Path|None=None):
        self.data = data
        self.trend = None
        self.residuals = None
        self.optimal_stl_df = None
        self.pvalues_df = None
        self.dirpath = dirpath
        
        # Define output filenames
        self.save = False
        if dirpath:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            self.save = True
            
            self.p_values_fpath = dirpath / "stl_p_values.csv" 
            self.optimal_stl_info_fpath = dirpath / "optimal_stl.csv"
            self.trends_fpath = dirpath / "trends.csv"
            self.residuals_fpath = dirpath / "residuals.csv"
            self.stationary_ts_fpath = dirpath / "stationary.csv"
            self.non_stationary_ts_fpath = dirpath / "non_stationary.csv"

    def reset(self):
        """
        Resets values of instance attributes
        without the need of re-initialization.
        """
        self.data = self.data
        self.trend = None
        self.residuals = None
        self.optimal_stls = None
        self.pvalues_df = None

    def _stl(self, y: pd.Series, tw: int) -> Tuple[pd.Series, pd.Series]:
        """
        Wrapper for the statsmodels STL functionality.

        Initializes an STL instance, fits it, extracts
        the trend and from there computes the residuals.

        Parameters
        ----------
        y : pandas.Series
            Pandas.Series containing the time series to be
            analyzed. Make sure the row index of y has a
            periodicity or is of a data type that allows
            for the inferral of a periodicity.

        tw : int
            Smoothing window length for LOESS in STL.

        Returns
        -------
        trend : pandas.Series
            Trend of input-time series y as extracted by
            STL.
            
        resid : pandas.Series
            Residuals remaining after subtracting the
            trend form the input-time series y.
        """
        stl = seasonal.STL(y, trend=tw)
        decomposition = stl.fit()
        trend = decomposition.trend
        resid = y-decomposition.trend
        return trend, resid

    def _optimize_trend(self, y: pd.Series, name: str) -> Tuple[pd.Series, int, float]:
        """
        Compute optimal smoothing window length for
        STL-based time series decomposition.

        Iterate over a list of smoothing window lengths.
        For each window length, perform an STL-based
        time series decomposition and compute an augmented
        Dickey-Fuller (ADF) test for the residuals. The
        smoothing window length minimizing this p-value (i.e.
        maximizing the confidence in the residuals being
        stationary) is considered the optimal smoothing window
        length.

        Parameters
        ----------
        y : pandas.Series
            Pandas.Series containing the time series to be
            analyzed. Make sure the row index of y has a
            periodicity or is of a data type that allows
            for the inferral of a periodicity.
            
        name : str
            Name of the time series

        Returns
        -------
        pval_series : pandas.Series
            Pandas.Series containing the p-values of the
            ADF test for a given metric and for all the 
            tested smoothing window lengths.
            
        tw_opt : int
            Smoothing window length found to be optimal with
            respect to the p-value of the ADF test.
        
        p_opt : float
            Optimal (minimal) p-value found among all ADF
            tests.
        """
        # Initialize variables 
        p_vals = []
        p_opt = np.inf
        tw_opt = None

        # Iterate over possible smoothing window lengths
        tw_grid = range(13, 501, 2)
        start = dt.now()
        for tw in tw_grid:
            trend, resid = self._stl(y, tw)
            adf=adfuller(resid)
            p = adf[1]

            # Update optimal values
            if p < p_opt:
                p_opt = p
                tw_opt = tw
                
            p_vals.append(p)
        end = dt.now()
        elapsed = end - start
        print(f"Trend for metric {name} optimized in {elapsed.seconds}s: (tw_opt = {tw_opt}, p_opt = {p_opt})", end="\r")

        pval_series = pd.Series(p_vals, index=tw_grid, name=name)

        return pval_series, tw_opt, p_opt
            
    def decompose(self,
                  trend_window: int|None=None,
                  optimize: bool=True) -> pd.DataFrame:
        """
        Decompose time series into trend and residual.

        Each time series is decomposed individually using STL.
        Specifically, the trend is estimated using LOESS (locally
        estimated scatterplot smoothing). The smoothing window
        is treated as a hyperparameter, which can be optimized.
        The optimal smoothing window is defined as the window
        that leads to the maximal confidence in the stationarity
        of the remaining trend.
        **REMARK:** This optimality definition is based on the
        assumption that the trend of all time series is stationary.

        Optionals
        ---------
        trend_window : int [default: None]
            Length of smoothing window for trend inference

        optimize : bool [default: True]
            Whether or not the trend smoothing window
            should be optimized.

        dirpath : Path [default: None]
            Directory path to where the result files
            should be written. If None results will
            not be written to disk at all.

        Returns
        -------
        pvalues_df : pd.DataFrame
            DataFrame containing p-value of augmented Dickey-
            Fuller test performed for estimated trend given
            the smoothing window length indicated by column
            header.

        Raises
        ------
        ValueError
            If optimize is set to False and trend_window
            is None.
        """
        self.trend = pd.DataFrame(index = self.data.index, columns = self.data.columns)
        self.residuals = pd.DataFrame(index = self.data.index, columns = self.data.columns)

        pvalues_list = []
        optimal_stl_info = []
        start = dt.now()

        for colcntr, col in enumerate(self.data.columns):
            print(f"Optimizing metric {colcntr}/{len(self.data.columns)}", end="\r")
            metric = self.data[col]
            y = metric.dropna()
            x = y.index

            # OPTIMIZATION
            # ============
            opt_tw = None
            if optimize:
                # Find optimal trend such that confidence in stationarity of remaining trend is maximal
                opt_series, opt_tw, opt_p = self._optimize_trend(y, col)
                pvalues_list.append(opt_series)
                optimal_stl_info.append({"metric": col,
                                         "optimal p-value": opt_p,
                                         "optimal smoothing window length": opt_tw
                                        })
            else:
                if trend_window is None:
                    raise ValueError("trend_window must be set when optimize = False.")
                opt_tw = trend_window
            
            # Decompose time series
            trend, resid = self._stl(y, opt_tw)
            
            self.trend.loc[x,col] = trend
            self.residuals.loc[x,col] = resid
        end = dt.now()
        elapsed = end - start
        print(f"Optimization completed in {elapsed.seconds}s.")

        self.trend.index.name = "date"
        self.residuals.index.name = "date"
        
        self.pvalues_df = pd.DataFrame(pvalues_list)
        self.pvalues_df.index.name = "metric"
        self.optimal_stl_df = pd.DataFrame(optimal_stl_info)

        if self.save:
            self.trend.to_csv(self.trends_fpath)
            self.residuals.to_csv(self.residuals_fpath)
            self.pvalues_df.to_csv(self.p_values_fpath)
            self.optimal_stl_df.to_csv(self.optimal_stl_info_fpath)
                    
        return self.residuals

    def test_stationarity(self, 
                          data: pd.DataFrame,
                          alpha: float=0.05,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tests stationarity for each out of a number of
        time series. 
        
        The pandas.DataFrame 'data' is expected to contain
        a number of different time series, each one
        corresponding to a column of the DataFrame. This
        function checks for each column, whether the
        corresponding time series is stationary or not by
        conducting an augmented Dickey-Fuller (ADF) test for 
        each time series separately.
        
        All the stationary time series are collected into
        a new pandas.DataFrame and so are the non-stationary
        ones. A 2-tuple with these two new pandas.DataFrames
        forms the output of this method.

        Parameters
        ----------
        data : pandas.DataFrame
            Pandas.DataFrame containing the different time
            series in its columns.
        
        Optionals
        ---------
        alpha : int [default: 0.05]
            Significance level used to decide whether or not
            a time series is stationary based on the p-value
            of the ADF test.

        Returns
        -------
        stationary_df : pandas.DataFrame
            Pandas.DataFrame containing all the metrics
            whose time series are stationary. For each metric
            (row) this table lists the ADF test statistic,
            the corresponding p-value, and the test statistic
            thereshold for the given signifance level alpha.
            
        non_stationary_df : pandas.DataFrame
            Pandas.DataFrame containing all the metrics
            whose time series are non-stationary. For each metric
            (row) this table lists the ADF test statistic,
            the corresponding p-value, and the test statistic
            thereshold for the given signifance level alpha.
        """
        non_stationary = []
        stationary = []
        
        for c in data.columns:
            adf_result=adfuller(data.loc[:,c].dropna())

            # rename components of adf_result variable for
            # better readability
            statistic = adf_result[0]  # value of test statistic
            p = adf_result[1]  # p value
            threshold = adf_result[4][f"{int(alpha*100)}%"]  # threshold for test statistic for given significance level alpha

            # Check stationarity & significance
            is_stationary = statistic<threshold
            is_significant = p<alpha
            
            if is_significant and is_stationary:
                stationary.append({"metric": c, 
                                   "test": statistic, 
                                   "p": p,
                                   f"max(test, p={int(alpha*100)}%)": threshold
                                  }
                                 )
            else:
                non_stationary.append({"metric": c, 
                                       "test": statistic, 
                                       "p": p, 
                                       f"max(test, p={int(alpha*100)}%)": threshold
                                      }
                                     )
        
        print(f"# non-stationary time series: {len(non_stationary)}")
        print(f"# stationary time series: {len(stationary)}")

        stationary_df = pd.DataFrame(stationary)
        non_stationary_df = pd.DataFrame(non_stationary)

        if len(stationary_df)>0:
            stationary_df = stationary_df.set_index("metric")
        if len(non_stationary_df)>0:
            non_stationary_df = non_stationary_df.set_index("metric")

        if self.save:
            stationary_df.to_csv(self.stationary_ts_fpath)
            non_stationary_df.to_csv(self.non_stationary_ts_fpath)
        
        return stationary_df, non_stationary_df