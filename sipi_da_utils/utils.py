# Std lib imports
from typing import Tuple, Dict, List
from itertools import combinations
from pathlib import Path
from datetime import datetime as dt

# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationAnalysis(object):
    """
    Perform correlation analysis of stationary
    time series data.

    Assumes that input data represent stationary
    time series data (i.e. no de-trending or other
    time series decomposition is performed).

    Parameters
    ----------
    data : pandas.DataFrame

    Optional Parameters
    -------------------
    id2name_map : pandas.DataFrame [default: None]
        DataFrame with Monet2030 metric IDs
        as index and a column "metric_name"
        containing the corresponding metric name.

    Attributes
    ----------
    data : pandas.DataFrame
        See parameter "data".

    Methods
    -------
    cross_corr(lag: int=0, verbosity: bool=0) -> pandas.DataFrame
    max_abs_corr() -> pandas.DataFrame
    plot_corr_heatmap(df: pandas.DataFrame, 
                      title: str,
                      fpath: Path=None
                      ) -> None
    drop_strong_correlations(corrmat: pandas.DataFrame,
                             threshold: float,
                             id2name_map: pandas.DataFrame|None=None,
                             verbose: int=0,
                             fpath_corr=None
                             ):
    """
    def __init__(self, data: pd.DataFrame, id2name_map: pd.DataFrame|None = None) -> pd.DataFrame:
        self.data = data
        self.id2name_map = id2name_map
        
    def cross_corr(self, lag: int = 0, verbosity: int = 0) -> pd.DataFrame:
        """
        Compute correlation between a dataframe and
        a lagged version of itself. If lag = 0, the
        result is equivalent to the usual correlation
        matrix.

        Optional Parameters
        -------------------
        lag : int [default: 0]
            Shift in the number of periods used in
            the second copy of the input matrix:
            X = self.data
            Y = X.shifted(periods=lag)

        verbosity : int [default: 0]
            Verbosity level governing the verbosity 
            of the output to standard out.

        Returns
        -------
        corr_df : pandas.DataFrame
            Dataframe containing the computed
            correlation matrix.
        """
        # Extract value arrays from dataframes
        X = self.data.values
        Y = self.data.shift(periods=lag).values
        
        # Masks for non-NaN (1 if valid, 0 if NaN)
        mask_X = (~np.isnan(X)).astype(float)
        mask_Y = (~np.isnan(Y)).astype(float)
        
        # Fill NaNs with 0 for efficient computation
        X_filled = np.nan_to_num(X)
        Y_filled = np.nan_to_num(Y)
        
        # Pairwise counts of overlapping non-NaNs
        n_ij = mask_X.T @ mask_Y  # (n1, n2) matrix
        
        # Sums for X, Y, XY, X², Y² over overlapping segments
        Sx = X_filled.T @ mask_Y    # Sum of X in overlapping segments
        Sy = mask_X.T @ Y_filled    # Sum of Y in overlapping segments
        Sxy = X_filled.T @ Y_filled # Sum of X*Y in overlapping segments
        Sx2 = (X_filled**2).T @ mask_Y  # Sum of X² in overlapping segments
        Sy2 = mask_X.T @ (Y_filled**2)  # Sum of Y² in overlapping segments
        
        # Initialize matrices for numerator and denominators
        numerator = np.full_like(n_ij, np.nan, dtype=float)
        denom_x = np.full_like(n_ij, np.nan, dtype=float)
        denom_y = np.full_like(n_ij, np.nan, dtype=float)
        
        # Compute only where n_ij > 0 (avoid division by zero)
        valid = n_ij > 0
        numerator[valid] = Sxy[valid] - (Sx[valid] * Sy[valid]) / n_ij[valid]
        denom_x[valid] = Sx2[valid] - (Sx[valid]**2) / n_ij[valid]
        denom_y[valid] = Sy2[valid] - (Sy[valid]**2) / n_ij[valid]
        
        # Ensure non-negative for square roots
        denom_x = np.sqrt(np.maximum(denom_x, 0))
        denom_y = np.sqrt(np.maximum(denom_y, 0))
        denom = denom_x * denom_y
        
        # Compute correlation (handle division by zero)
        corr = np.full_like(numerator, np.nan, dtype=float)
        nonzero_denom = (denom != 0)
        corr[nonzero_denom] = numerator[nonzero_denom] / denom[nonzero_denom]
        
        # Set correlations to NaN where n_ij < 2 (insufficient samples)
        corr[n_ij < 10] = np.nan

        self.corr_df = pd.DataFrame(corr, index=self.data.columns, columns=self.data.columns)
        return self.corr_df

    def max_abs_corr(self) -> pd.DataFrame:
        """
        Computes maximum absolute correlation
        between self.data and self.data.shifted
        for all possible lags.

        Returns
        -------
        agg_corrmat : pd.DataFrame
            Correlation matrix containing the 
            maximum absolut correlation value
            between any two pairs of columns
            maximized over all lags.
        """
        # Compute the cross correlations between self.data and all
        # shifted versions of itself. The upper bound for the number
        # of lags is given by the number of rows in self.data. Any
        # larger lag will always result in an all-nan correlation
        # matrix as the there will not be any overlapping elements
        # between X and X.shifted(period>n_rows).
        n_rows = self.data.shape[0]
        corrmat_list = [self.cross_corr(lag=l) for l in range(n_rows)]

        # Create a 3D numpy array from the result. The 3rd dimension
        # has length equal to n_rows.
        self.corrmat_stack = np.dstack([cc.values for cc in corrmat_list])

        # Now we aggregate along that new 3rd dimension. We perform
        # to aggragations: min and max
        cols = self.data.columns
        max_corr = pd.DataFrame(np.nanmax(self.corrmat_stack,axis=-1),
                                index=cols,
                                columns=cols
                               )
        
        min_corr = pd.DataFrame(np.nanmin(self.corrmat_stack,axis=-1),
                                index=cols,
                                columns=cols
                               )

        # Ultimately we are interested in correlation in absolute terms.
        # Therefore, if abs(min)>max, then we actually focus on abs(min)
        # and don't really care about max. The next code line combines
        # the two dataframes max_corr and min_corr.abs() in a single 
        # dataframe where each element is the maximum of the corresponding 
        # elements in max_corr and min_corr.abs.
        self.agg_corrmat = max_corr.combine(min_corr.abs(), 
                                            func=np.maximum)\
                                   .round(5)

        return self.agg_corrmat

    def plot_corr_heatmap(self, 
                          df: pd.DataFrame, 
                          title: str, 
                          fpath: Path|None = None):
        """
        Plots heatmap of input dataframe df.

        Parameters
        ----------
        df : pandas.DataFrame
            The data which shall be visualized
            as a heatmap.

        title : str
            Title of the heatmap plot

        Optional Parameters
        -------------------
        fpath : pathlib.Path [default: None]
            Path to file to which the heatmap plot
            should be written.
        """
        if self.id2name_map:
            idx_name_map = self.id2name_map["metric_name"].to_dict()
            df = df.rename(idx_name_map, axis=1).rename(idx_name_map, axis=0)
            
        fig, ax = plt.subplots(figsize=(10,9))    
        sns.heatmap(df, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)
        plt.tight_layout()

        fig_title = "./" + title
    
        if fpath:
            fig.savefig(fpath)

    def _sort_corrmat_by_variances(self, 
                                   corrmat: pd.DataFrame,
                                   ascending: bool=False
                                  ) -> pd.DataFrame:
        """
        Merges a column of metric variances with the
        input correlation matrix and sorts the rows of
        the result by the variance column.

        Parameters
        ----------
        corrmat : pandas.DataFrame
            Correlation matrix
            
        Optional Parameters
        -------------------
        ascending : bool [default: False]
            Indicates sorting order of rows
            in correlation matrix.
            
        Retruns
        -------
        varsorted : pandas.DataFrame
            Correlation matrix with new, prepended
            variance column. Rows are sorted by variance
            column.
        """
        # Compute variances of values for each metric
        varsvec = self.data.var()
        
        # Next create a matrix with the variances of each metric in the first column
        # and the covariances/correlations in all the subsequent columns.
        # Sort the resulting matrix by variance in descending order.
        # This step is performed to sort the rows of the correlation matrix by
        # the variances of the respective metrics.
        varcovar = varsvec.to_frame()\
                          .merge(corrmat, left_index=True, right_index=True)\
                          .rename({0: "variance"}, axis=1)\
                          .sort_values(by="variance", ascending=ascending)
        
        # Now that the correlation matrix is sorted, we can drop the variance column again
        varsorted = varcovar.drop("variance", axis=1)

        return varsorted 

    def drop_strong_correlations(self, 
                                 corrmat: pd.DataFrame,
                                 threshold: float,
                                 verbose: int=0,
                                 fpath_corr: Path|None=None
                                ) -> Tuple[List[str], Dict[str, pd.Series]]:
        """
        Filter which indices to keep and which ones to drop.

        This method prunes the metrics listed in corrmat
        based on their mutual correlations. It does so
        in order of decending variance of each metric, i.e.
        the code starts looking at the highest-variance
        metric (let's call it "M1") and prunes all metrics
        that are correlated with M1 more than threshold.
        From the surviving (not pruned) metrics again
        look at the highest-variance one and repeat the
        process until there are no more metrics that
        have not yet been pruned or visited.
        
        Parameters
        ----------
        corrmat : pandas.DataFrame
            Correlation matrix.

        threshold : float
            Correlation threshold. Any correlation
            between two metrics higher than threshold
            is considered to be an indicator that the
            two metrics are redundant.

        Optional Parameters
        -------------------
        verbose : int [default: 0]
            Verbosity level for standard out.

        fpath_corr : pathlib.Path [default: None]
            Path to file to which the matrix
            "correlation_xlsx" should be written.

        Returns
        -------
        to_keep: List[str]
            List of metric IDs or names that survive
            the filtering.        
        
        correlation_xlsx : Dict[str, pandas.Series]
            Dictionary with key-value pairs, where the
            key is a metric ID or name and the value is
            a pandas.Series containing all other metrics
            that are correlated more than the threshold
            with the key.

        Raises
        ------
        ValueError
            If a metric is tried to be dropped that was already
            dropped earlier. This indicates a problem in the code.

        ValueError
            If kept_metrics != to_keep, i.e. if the length of the 
            output list to_keep is not equal to the number of metrics
            surviving in the correlation matrix.
            
        ValueError
            If set(to_keep).intersection(set(to_drop)) != set(), i.e.
            if a metric is both kept and being dropped. That's impossible!

        ValueError
            If len(to_keep)+len(to_drop) != n_start, i.e. if the sum
            of the kept and dropped metrics is not equal to all the metrics
            we started with. This means that either we have missed a metric
            or we looked at one or more metrics more than just once.
        """

        # Setup
        varsorted_corrmat = self._sort_corrmat_by_variances(corrmat, ascending=False)
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
        
            # Round correlation values to two decimals for better
            # readability
            to_drop_ser = to_drop_ser.apply(lambda x: round(x,3))
            
            if verbose>0:
                # Print all metrics that should be dropped because they
                # are correlated more strongly than threshold (in absolute
                # value-terms) with the current test metric and are therefore
                # considered redundant.
                print(f"{to_drop_ser.name} (var = {curr_corrs.sort_values(ascending=False).iloc[i].round(3)}): {to_drop_ser.to_dict()}")

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
            n_removed = len(to_drop_ser)
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
                print(varsorted_corrmat.shape)
                print(f"Remaining: {n_remaining} metrics")
                print()

            # Update index
            i += 1

        if self.id2name_map is not None:
            named_dict = dict()
            for k, v in correlation_xlsx.items():
                named_key = self.id2name_map.loc[k, "metric_name"]
                named_values = v
                named_values.index = [self.id2name_map.loc[mid, "metric_name"] for mid in named_values.keys()]
                named_values.name = named_key
                named_dict[named_key] = named_values

            correlation_xlsx = named_dict

        if fpath_corr:
            with pd.ExcelWriter(fpath_corr) as writer:
                for sheet_name, ser in correlation_xlsx.items():
                    sheet_name = sheet_name.replace("[","_")\
                                           .replace("]","_")\
                                           .replace("/","_or_")\
                                           .replace(":","_") 
                    ser.to_frame().to_excel(writer, sheet_name=sheet_name, index=True)

        # Sanity checks
        # =============
        kept_metrics = [c for c in varsorted_corrmat.columns]
        if kept_metrics != to_keep:
            raise ValueError("kept_metrics != to_keep")
        if set(to_keep).intersection(set(to_drop)) != set():
            raise ValueError("set(to_keep).intersection(set(to_drop)) != set()")
        if len(to_keep)+len(to_drop) != n_start:
            if verbose>0:
                print(f"len(to_keep)={len(to_keep)}")
                print(f"len(to_drop)={len(to_drop)}")
                print(f"n_start = {n_start}")
            raise ValueError("len(to_keep)+len(to_drop) != n_start")

        return to_keep, correlation_xlsx