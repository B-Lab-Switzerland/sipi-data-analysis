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


def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the values across rows (i.e. within
    any given column) linearly.
    """
    df_interp = df.astype(float).interpolate("cubicspline",axis="rows", limit_area="inside")
    return df_interp

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

class DataImputer(object):
    """
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.measurements = dict()
        self.predictions = dict()
        
    def fit_gp(self):
        """
        """
        min_year = 2025
        max_year = 1900
        for counter, metric in enumerate(self.df.columns):
            print(f"{counter+1}/{len(self.df.columns)}", end="\r")
            
            # Measurements
            metr_series = self.df[metric].dropna()
            X = np.array([year for year in metr_series.keys()]).reshape(-1, 1)  # GP needs 2D inputs
            y = np.array([val for val in metr_series.values]).ravel()

            # Get year range [min_year, max_year]
            if X.min() < min_year:
                min_year = X.min()
            if X.max() > max_year:
                max_year = X.max()
        
            # Define a kernel: RBF (smoothness) + WhiteKernel (noise)
            kernel = 1.0 * Matern(length_scale=3.3, nu=1.5)
        
            # Fit GP
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-10, normalize_y=True)
            gp.fit(X, y)
        
            # Predict (grids have different ranges for different metrics, but all have equal spacing)
            X_common = np.arange(X.min(), X.max() + 1.0/12, 1.0/12).reshape(-1, 1)
            y_pred, y_std = gp.predict(X_common, return_std=True)
        
            # Gather info
            self.measurements[metric] = [X,y]
            self.predictions[metric] = [X_common, y_pred, y_std]

        # Compile the results into a single dataframe
        full_range = [np.round(i,2) for i in np.arange(min_year, max_year + 1.0/12, 1.0/12)]
        self.gp_means = pd.DataFrame(index=full_range, 
                                     columns=self.df.columns
                                     )

        self.gp_stds = pd.DataFrame(index=full_range, 
                                    columns=self.df.columns
                                   )

        # Check that key/column names of self.df and self.measurements are aligned
        assert set([metric for metric in self.predictions.keys()]) == set([col for col in self.df.columns])
        
        # Fill data into self.gp_means column by column
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
           

class TSAnalyzer(object):
    """
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.trend = None
        self.residuals = None

    def reset(self):
        self.data = self.data
        self.trend = None
        self.residuals = None

    def _stl(self, y, tw):
        stl = seasonal.STL(y, trend=tw, seasonal=3)
        decomposition = stl.fit()
        trend = decomposition.trend
        resid = y-decomposition.trend
        return trend, resid

    def _optimize_trend(self, y, name):
        # Initialize variables 
        p_vals = []
        p_opt = np.inf
        tw_opt = None

        # Iterate
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

        return pd.Series(p_vals, index=tw_grid, name=name), tw_opt, p_opt
            
        
    def decompose(self, trend_window=None, optimize=True):
        """
        """
        self.trend = pd.DataFrame(index = self.data.index, columns = self.data.columns)
        self.residuals = pd.DataFrame(index = self.data.index, columns = self.data.columns)

        optimality_list = []
        start = dt.now()
        for col in self.data.columns:
            metric = self.data[col]
            y = metric.dropna()
            x = y.index

            # OPTIMIZATION
            # ============
            optimal_tw = None
            if optimize:
                # Find optimal trend such that confidence in stationarity of remaining trend is maximal
                opt_series, optimal_tw, opt_p = self._optimize_trend(y, col)
                optimality_list.append(opt_series)
            else:
                if trend_window is None:
                    raise ValueError("trend_window must be set when optimize = False.")
                optimal_tw = trend_window
            
            # Decompose time series
            trend, resid = self._stl(y, optimal_tw)
            
            self.trend.loc[x,col] = trend
            self.residuals.loc[x,col] = resid
        end = dt.now()
        elapsed = end - start
        print(f"Optimization completed in {elapsed.seconds}s.")

        return pd.DataFrame(optimality_list)

    def test_stationarity(self, alpha: float=0.05, data: pd.DataFrame=None):
        """
        """
        non_stationary = []
        stationary = []
        
        if data is None:
            if self.residuals:
                print("Checking stationarity for residuals (data = self.residuals)...")
                data = self.residuals
            else:
                print("Checking stationarity for raw/interpolated data (i.e. not for residuals)...")
                data = self.data
            
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
                stationary.append({"metric": c, "test": statistic, "p": p, f"max(test, p={int(alpha*100)}%)": threshold})
            else:
                non_stationary.append({"metric": c, "test": statistic, "p": p, f"max(test, p={int(alpha*100)}%)": threshold})
        
        print(f"# non-stationary time series: {len(non_stationary)}")
        print(f"# stationary time series: {len(stationary)}")

        return (pd.DataFrame(stationary), pd.DataFrame(non_stationary))

    def optimize_trend(self, trend_window_grid, alpha):
        """
        Naïvely optimize the trend window length by maximizing
        the ratio of stationary-to-non-stationary time series.
        """
        ratios = []
        optimal_tw = min(trend_window_grid)
        optimal_ratio = 0
        for tw in trend_window_grid:
            self.detrend(trend_window=tw)
            stat, non_stat = self.test_stationarity(alpha=alpha, data=self.residuals)
            ratio = stat/non_stat
            ratios.append(ratio)

            # update optimum
            if ratio > optimal_ratio:
                optimal_ratio = ratio
                optimal_tw = tw
                
            self.reset()
        
    
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