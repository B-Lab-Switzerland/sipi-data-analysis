# Std lib imports
from pathlib import Path
from typing import List, Set, Dict, Tuple

# 3rd party imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

# Local imports
from sipi_da_utils import utils
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
        #all_metrics = [metric for metric in self.data["raw"].columns]
        #pruned_metrics = set(all_metrics) - set(kept_metrics)
        #pruned_metrics_df = metric2capital_map.loc[metric2capital_map.index.isin(pruned_metrics)]

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

        if data_fpath is not None:
            n_metrics_per_cap.to_csv(data_fpath) 

        # Generate plot of result
        fig, ax = plt.subplots(figsize=(7,4))
        ax = self._plot_number_of_metrics_per_capital(n_metrics_per_cap, ax)

        if plot_fpath is not None:
            fig.savefig(plot_fpath)

        plt.show()

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

        fig, ax = plt.subplots(figsize=(8,3))
        ax = self._plot_number_of_irrelevant_metrics_per_capital(irrelevant_metrics_per_capital, ax)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(const.n_irrelevant_by_capital_plot_fpath)
        plt.show()

        return irrelevant_metrics_per_capital