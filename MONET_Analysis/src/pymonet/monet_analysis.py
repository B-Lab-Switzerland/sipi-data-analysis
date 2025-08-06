# Std lib imports
from pathlib import Path
from typing import List, Set, Dict, Tuple

# 3rd party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

# Local imports
from sipi_da_utils import utils, plot
from pymonet import monet_consts as const


class MonetAnalyzer(object):
    def __init__(self, 
                 results_dict: Dict[str, pd.DataFrame] 
                ):
        self.data = results_dict
        self.capital = self._get_capitals()
        self.metric2capital_map = self.data['metric_id2name_map']

    def _get_capitals(self) -> Set[str]:
        """
        Extract all capitals from the
        metric2capital_map.

        Parameters
        ----------
        None

        Returns
        -------
        all_caps : Set[str]
            A set containing all capitals
        """
        all_caps = set([c for c in self.data['metric_id2name_map']["capital - primary"]])
        return all_caps

    def _plot_number_of_metrics_per_capital(self, n_metrics_per_cap: pd.DataFrame, ax: Axes) -> Axes:
        """
        Create plot showing the number of metrics
        per capital both before and after data
        cleaning.

        Parameters
        ----------
        n_metrics_per_cap: pandas.DataFrame
            Table containing the relevant counts
        
        ax: Axes
            Axes on whose canvas the plot shall
            be drawn

        Returns
        -------
        ax: Axes
            Axes on whose canvas the plot was
            drawn
        """
        n_metrics_per_cap.plot(kind="bar", ax=ax)
        ax.grid(True)
        ax.set_title("Number of Metrics per Capital")
        ax.set_xticks([0,1,2,3], n_metrics_per_cap.index, rotation=0)
        ax.set_ylabel("Count")
        return ax

    def _number_of_metrics_per_capital__compute(self):
        """
        """
        metrics_meta_table = self.metric2capital_map[self.metric2capital_map.index.str.endswith("metr")]

        kept_metrics = list(self.data["clean"].columns)
        kept_metrics_df = self.metric2capital_map.loc[self.metric2capital_map.index.isin(kept_metrics)]
        
        
        # Count the number of metrics per capital BEFORE data cleaning
        n_metrics_per_cap_all = metrics_meta_table.groupby("capital - primary")\
                                                  .agg({"metric_name": "count"})

        # Count the number of metrics per capital AFTER data cleaning
        kept_metrics_idx = metrics_meta_table.index.isin(kept_metrics)
        n_metrics_per_cap_cleaned = metrics_meta_table.loc[kept_metrics_idx,:]\
                                                      .groupby("capital - primary")\
                                                      .agg({"metric_name": "count"})
        
        # Join the two together into a single table
        n_metrics_per_cap_all = n_metrics_per_cap_all.rename({"metric_name": "before cleaning"}, axis=1)
        n_metrics_per_cap_cleaned = n_metrics_per_cap_cleaned.rename({"metric_name": "after cleaning"}, axis=1)
        n_metrics_per_cap = n_metrics_per_cap_all.join(n_metrics_per_cap_cleaned)

        return n_metrics_per_cap

    def _number_of_metrics_per_capital__plot(self, data):
        """
        """
        # Generate plot of result
        fig, ax = plt.subplots(figsize=(7,4))
        ax = self._plot_number_of_metrics_per_capital(n_metrics_per_cap, ax)
        return fig, ax
        
    def number_of_metrics_per_capital(self, 
                                      data_fpath: Path|None = None, 
                                      plot_fpath: Path|None = None
                                     ):
        """
        Count the number of metrics per capital
        both before and after data cleaning.

        Optional Parameters
        -------------------
        data_fpath : Path [default: None]
            Path to the file to which the resulting
            table with counts should be written.
            Writing is desabled when data_fpath is None.

        plot_fpath : Path [default: None]
            Path to the file to which the resulting
            plot with counts should be written.
            Writing is desabled when plot_fpath is None.
        """
        # Compute
        # -------
        n_metrics_per_cap = self._number_of_metrics_per_capital__compute()
        if data_fpath is not None:
            n_metrics_per_cap.to_csv(data_fpath) 

        # Plot
        # ----
        fig, ax = self._number_of_metrics_per_capital__plot(n_metrics_per_cap)
        if plot_fpath is not None:
            fig.savefig(plot_fpath)
            
        return n_metrics_per_cap

    def _plot_number_of_sparse_metrics_per_capital(self,
                                                   sparse_metrics_per_capital: pd.DataFrame,
                                                   ax: Axes
                                                  ) -> Axes:
        """
        """
        sparse_metrics_per_capital.plot(kind="bar", ax=ax, legend=False)
        ax.grid(True)
        ax.set_xlabel("capital")
        ax.set_ylabel("count")
        ax.set_title("Number of removed metrics per capital due to insufficient data availability")

        return ax
        
    def number_of_sparse_metrics_per_capital(self, 
                                             data_fpath: Path|None = None, 
                                             plot_fpath: Path|None = None
                                            ):
        """
        """
        sparse_metrics = list(self.data["sparse_cols"]["sparse columns (<10 data points)"].values)
        sparse_metrics_df = self.metric2capital_map.loc[self.metric2capital_map.index.isin(sparse_metrics)]

        sparse_metrics_df = sparse_metrics_df.join(self.data["raw"].count(axis=0)
                                             .to_frame()
                                             .rename({0: "count"}, axis=1),
                                             how="left"
                                            )
        sparse_metrics_df = sparse_metrics_df.sort_values(by="count", ascending=False)
        sparse_metrics_df.to_csv(const.sparse_metrics_analysis_fpath) 
        sparse_metrics_df.head()

        sparse_metrics_per_capital = sparse_metrics_df.groupby("capital - primary").agg({"count": "count"})

        if data_fpath is not None:
            sparse_metrics_per_capital.to_csv(data_fpath)

        # Create plot
        fig, ax = plt.subplots(figsize=(8,3))
        ax = self._plot_number_of_sparse_metrics_per_capital(sparse_metrics_per_capital, ax)
        plt.xticks(rotation=0)
        plt.tight_layout()

        if plot_fpath is not None:
            fig.savefig(plot_fpath)
            
        plt.show()

        return sparse_metrics_per_capital

    def _plot_number_of_irrelevant_metrics_per_capital(self,
                                                       irrelevant_metrics_per_capital: pd.DataFrame,
                                                       ax: Axes
                                                      ) -> Axes:
        """
        """
        irrelevant_metrics_per_capital.plot(kind="bar", ax=ax, legend=False)
        ax.grid(True)
        ax.set_xlabel("capital")
        ax.set_ylabel("count")
        ax.set_title("Number of metrics irrelevant to agenda2030 (per capital)")
        
    def number_of_irrelevant_metrics_per_capital(self, 
                                                 data_fpath: Path|None = None, 
                                                 plot_fpath: Path|None = None
                                                ):
        """
        """
        irrelevant_metrics = list(self.data["irrelevant_metrics"].columns)
        irrelevant_metrics_df = self.metric2capital_map.loc[self.metric2capital_map.index.isin(irrelevant_metrics)]
        
        irrelevant_metrics_df = irrelevant_metrics_df.join(self.data["raw"].count(axis=0)
                                                   .to_frame()
                                                   .rename({0: "count"}, axis=1),
                                                   how="left"
                                                  )
        irrelevant_metrics_df = irrelevant_metrics_df.sort_values(by="count", ascending=False)
        irrelevant_metrics_df.to_csv(const.irrelevant_metrics_analysis_fpath)
        irrelevant_metrics_df.head()

        irrelevant_metrics_per_capital = irrelevant_metrics_df.groupby("capital - primary").agg({"count": "count"})

        if data_fpath is not None:
            irrelevant_metrics_per_capital.to_csv(data_fpath)

        fig, ax = plt.subplots(figsize=(8,3))
        ax = self._plot_number_of_irrelevant_metrics_per_capital(irrelevant_metrics_per_capital, ax)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        if plot_fpath is not None:
            fig.savefig(plot_fpath)
            
        plt.show()

        return irrelevant_metrics_per_capital

    def _close_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills in missing years in columns of df.

        The argument df is assumed to have integer-typed
        years as column headers. If any years are missing
        between the earliest and the latest year, these
        gaps are filled.

        Parameters
        ----------
        df : pandas.DataFrame
            Table whose column index is checked for
            time gaps and potentially modified.
            
        Returns
        -------
        df : pandas.DataFrame
            The same as input df but with potential time
            gaps filled.
        """
        
        df = df.loc[:,~df.columns.duplicated()].copy()
        
        # Get existing years from column index (assumed datetime index)
        existing_years = [col for col in df.columns]
        full_year_range = list(range(min(existing_years), max(existing_years) + 1))
        
        # Identify missing years
        missing_years = [y for y in full_year_range if y not in existing_years]
        
        # Add missing columns with white color
        for year in missing_years:
            df[year] = np.nan * len(df)
        
        # Reorder columns chronologically
        df = df.reindex(sorted(df.columns), axis=1)
        
        df["capital"] = self.metric2capital_map.loc[df.index, "capital - primary"]

        return df

    @staticmethod
    def draw_data_availability_plot(self,
                                    df_plot: pd.DataFrame,
                                    metric_availability: pd.DataFrame, 
                                    axs: np.ndarray):
        """
        """
        # Left plot panel
        axs[0] = plot.visualize_data_availability_colored(df_plot,
                                                          "Year",
                                                          "Metric Name",
                                                          "MONET2030 metric data availability across time",
                                                          ax=axs[0]
                                                         )
        
        # Right plot panel
        axs[1] = plot.raw_data_availability_barchart(metric_availability,
                                                     "Number of measured data points per metric",
                                                     "Metric Name",
                                                     "Number of measured data points per metric",
                                                     ax=axs[1],
                                                     show_legend=False
                                                     )

        return axs

    def raw_data_availability(self,
                              data_fpath: Path|None = None, 
                              plot_fpath: Path|None = None
                             ):
        """
        """
        datapoint_counts = self.data["raw"].count()\
                                           .sort_values(ascending=False)\
                                           .to_frame()\
                                           .rename({0: "count"}, axis=1)
        
        metric_availability = datapoint_counts.join(self.metric2capital_map)
        metric_availability = metric_availability.dropna(subset=["capital - primary"])

        if data_fpath is not None:
            metric_availability.to_csv(data_fpath)

        # Transpose data for availability plot
        monet_trp = self.data["raw"].transpose()
        monet_trp = self._close_time_gaps(monet_trp)
        
        # PLOT #
        # ---- #
        df_plot = monet_trp.loc[metric_availability.index,:]
        
        fig,axs=plt.subplots(1,2, figsize=(17,30), sharey=True, gridspec_kw = {"wspace": 0.01})
        axs = draw_data_availability_plot(df_plot, metric_availability, axs)
        plt.tight_layout()
        fig.suptitle("Data availability", y=0.9, fontsize=18)

        if plot_fpath is not None:
            fig.savefig(plot_fpath, bbox_inches="tight")
        plt.show()

        return metric_availability