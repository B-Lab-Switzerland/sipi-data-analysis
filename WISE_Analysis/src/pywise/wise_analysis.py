# Std lib imports
from pathlib import Path
from typing import List, Set, Dict, Tuple, Iterable
from abc import ABC, abstractmethod

# 3rd party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

# Local imports
from sipi_da_utils import utils, plot
from pywise import wise_consts as const

class Analyzer(ABC):
    """
    Abstract base class for various kinds
    of data analysis tasks in the context
    of WISE.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        wise_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.
        
    Attributes
    ----------
    input: pandas.DataFrame
        See input parameter "data".
        
    metrics_meta_table : pandas.DataFrame
        Table containing meta information
        for each metric.
        
    id2name_map : dict
        A dictionary mapping the metric IDs
        to the corresponding metric names.
        
    capitallist = list
        A list containing all capitals:
        - human
        - social
        - natural
        - economic
        
    overwrite : bool
        See input parameter "overwrite".
        
    output : Any
        Main result of the data analysis
        step.
        
    additional_results : dict[str, Any]
        A dictionary containing additional
        results of the data analysis step.
    
    Abstract methods
    ----------------     
    _compute(self) -> pd.DataFrame|List[Any]
    _plot(self, df: pandas.DataFrame):       

    Concrete methods
    ----------------
    analyze() -> List[Any]
    """
    def __init__(self, 
                 data: pd.DataFrame,
                 overwrite: bool = False
                ):
        self.input = data
        self.metrics_meta_table = self._get_metrics_meta_table()
        self.id2name_map = self.metrics_meta_table["metric_name"].to_dict()
        self.capitallist = [cap for cap in self.metrics_meta_table["capital - primary"].unique()]
        self.overwrite = overwrite

        # Containers
        self.output = None
        self.additional_results = dict()

    def _get_metrics_meta_table(self) -> pd.DataFrame:
        """
        Read in meta data about each metric
        from file.

        Notice that the read-in file contains
        rows with some duplicated metric_ids. 
        This is because the read-in file contains
        a table resulting from a join of metric-level
        meta information and indicator-level meta
        information and because some metrics are
        related to several different indicators.
        As for the purposes of the analysis module
        we only need metric_IDs, names, descriptions,
        and capitals.

        Parameters
        ----------
        None
        
        Returns:
        --------
        metrics_meta_table : pandas.DataFrame
            Table containing the deduplicated metric-
            level meta-information.    
        """
        metrics_meta_table = pd.read_csv(const.wise_metatable_fname).set_index("metric_id")
        
        # **REMARK**
        # Some metrics are related to more than one indicator.
        # As a result, some metrics may be repeated in the 
        # metrics_meta_table, in which case the metric-specific
        # information is identical but the related indicator-
        # or observable-level information may differ. For instance
        # a given metric can be related to two different indicators
        # and thus two different sub-SDGs. Both these rows
        # would still have the same row index (metric ID). This
        # duplication can lead to problems down the road, i.e. we
        # need to perform a deduplication. But for this to work
        # we can only consider metric-level information such as
        # metric ID, metric name, and capital.
        metrics_meta_table = metrics_meta_table[["metric_name", 
                                                 "metric_description",
                                                 "capital - primary"
                                                ]].drop_duplicates()

        return metrics_meta_table
         
    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def _plot(self, df):
        pass

    def _save_data(self, data: dict, path: Path):
        """
        Save the data to disk at "path".

        The function checks if the file at
        "path" already exists and overwrites
        it only if self.overwrite = True.
        
        Parameters
        ----------
        data : pandas.DataFrame|dict
            Data to be written to disk

        path : pathlib.Path
            File path to where the data
            shall be written.

        Returns
        -------
        None
        """          
        if path.exists() and not(self.overwrite):
            print(f"File {path.as_posix()} already exists... not overwriting!")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.overwrite:
                print(f"Overwriting existing file {path.as_posix()}...")
            
            if isinstance(data, dict):
                assert path.as_posix().endswith(".xlsx")
                print(f"Writing file {path.as_posix()}...")
                with pd.ExcelWriter(path) as writer:
                    for name, df in data.items():
                        name = str(name).replace("[","_")\
                                   .replace("]","_")\
                                   .replace("/","_or_")\
                                   .replace(":","_") 
                        if isinstance(df, list):
                            df = pd.Series(df)
                            
                        df.to_excel(writer, sheet_name=name)
                print("-> Done!")

    def _save_figure(self, fig: Figure, path: Path, dpi: int=100):
        """
        Save the figure to disk at "path".

        The function checks if the figure file 
        at "path" already exists and overwrites
        it only if self.overwrite = True.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to be written to disk

        path : pathlib.Path
            File path to where the figure
            shall be written.

        Optional Parameters
        -------------------
        dpi : int [default: 100]
            "Dots per inch" (i.e. resolution)
            used when saving the figure.

        Returns
        -------
        None
        """
        assert path.as_posix().endswith(".png") or path.as_posix().endswith(".pdf")
        
        if path.exists() and not(self.overwrite):
            print(f"Figure {path.as_posix()} already exists... not overwriting!")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.overwrite:
                print(f"Overwriting existing file {path.as_posix()}...")
            print(f"Writing file {path.as_posix()}...")
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            print("-> Done!")
            
    def analyze(self):
        """
        Execute the current analysis
        step.

        An analysis step consists of computing
        the respective analysis results and
        subsequently producing a plot based
        on these results.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._compute()
        self._plot()

        
class nMetricsPerCapital(Analyzer):
    """
    Analyze number of metrics per capital
    before and after data cleaning.

    Child of ABC "Analyzer".

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        wise_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.
        
    Attributes
    ----------
    input: pandas.DataFrame
        See input parameter "data".
        
    metrics_meta_table : pandas.DataFrame
        Table containing meta information
        for each metric.
        
    id2name_map : dict
        A dictionary mapping the metric IDs
        to the corresponding metric names.
        
    capitallist = list
        A list containing all capitals:
        - human
        - social
        - natural
        - economic
        
    overwrite : bool
        See input parameter "overwrite".
        
    output : Any
        Main result of the data analysis
        step.
        
    additional_results : dict[str, Any]
        A dictionary containing additional
        results of the data analysis step.

    Private Methods
    ---------------
    _compute() -> None
    _plot() -> None

    Public Methods
    --------------
    analyze() -> None
        Is defined in parent ABC "Analyzer"
    """
    def _compute(self):
        """
        Count the number of metrics per capital
        before and after cleaning.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        n_metrics_per_cap_dict = dict()
        for iso3, df in self.input.items():
            kept_metrics = list(df.columns)
            kept_metrics_df = self.metrics_meta_table.loc[self.metrics_meta_table.index.isin(kept_metrics)]
            
            
            # Count the number of metrics per capital BEFORE data cleaning
            n_metrics_per_cap_all = self.metrics_meta_table.groupby("capital - primary")\
                                                      .agg({"metric_name": "count"})
    
            # Count the number of metrics per capital AFTER data cleaning
            kept_metrics_idx = self.metrics_meta_table.index.isin(kept_metrics)
            n_metrics_per_cap_cleaned = self.metrics_meta_table.loc[kept_metrics_idx,:]\
                                                          .groupby("capital - primary")\
                                                          .agg({"metric_name": "count"})
            
            # Join the two together into a single table
            n_metrics_per_cap_all = n_metrics_per_cap_all.rename({"metric_name": "before cleaning"}, axis=1)
            n_metrics_per_cap_cleaned = n_metrics_per_cap_cleaned.rename({"metric_name": "after cleaning"}, axis=1)
            n_metrics_per_cap = n_metrics_per_cap_all.join(n_metrics_per_cap_cleaned)
            n_metrics_per_cap_dict[iso3] = n_metrics_per_cap.fillna(0).astype(int)

        # Make results available
        self.output = n_metrics_per_cap_dict
        
        # Write result to disk
        self._save_data(n_metrics_per_cap_dict, const.n_metrics_per_cap_fpath)

    def _plot(self):
        """
        Create plot showing the number of metrics
        per capital both before and after data
        cleaning.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for iso3, df in self.output.items():
            fig, ax = plt.subplots(figsize=(7,4))
            display(df)
            df.plot(kind="bar", ax=ax)
            ax.grid(True)
            ax.set_title(f"{iso3}: Number of Metrics per Capital")
            ax.set_xticks(range(len(df.index)), df.index, rotation=0)
            ax.set_xlabel("Capital")
            ax.set_ylabel("Count")
            self._save_figure(fig, const.n_metrics_per_cap_plot_fpath)
            plt.show()


class nSparseMetricsPerCapital(Analyzer):
    """
    Analyze number of sparse metrics per capital.
    A metric is considered sparse if there are
    less than 10 data points.

    Child of ABC "Analyzer".

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        wise_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.
        
    Attributes
    ----------
    input: pandas.DataFrame
        See input parameter "data".
        
    metrics_meta_table : pandas.DataFrame
        Table containing meta information
        for each metric.
        
    id2name_map : dict
        A dictionary mapping the metric IDs
        to the corresponding metric names.
        
    capitallist = list
        A list containing all capitals:
        - human
        - social
        - natural
        - economic
        
    overwrite : bool
        See input parameter "overwrite".
        
    output : Any
        Main result of the data analysis
        step.
        
    additional_results : dict[str, Any]
        A dictionary containing additional
        results of the data analysis step.

    Private Methods
    ---------------
    _compute() -> None
    _plot() -> None

    Public Methods
    --------------
    analyze() -> None
        Is defined in parent ABC "Analyzer"
    """
    def _compute(self):
        """
        Count the number of sparse metrics per 
        capital.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sparse_metrics_counts_dict = dict()
        sparse_metrics_names_dict = dict()
        for iso3, df in self.input.items():
            sparse_metrics = list(df["sparse columns (<10 data points)"].values)
            sparse_metrics_df = self.metrics_meta_table.loc[self.metrics_meta_table.index.isin(sparse_metrics)]
            sparse_metrics_names = sparse_metrics_df.groupby("capital - primary")\
                                                    .agg({"metric_name": lambda x: ", ".join(x)})
            sparse_metrics_df = sparse_metrics_df.groupby("capital - primary")\
                                                 .agg({"metric_name": "count"})\
                                                 .rename({"metric_name": "count"}, axis=1)\
                                                 .sort_values(by="count", ascending=False)
            sparse_metrics_counts_dict[iso3] = sparse_metrics_df
            sparse_metrics_names_dict[iso3] = sparse_metrics_names

        # Make results available
        self.output = sparse_metrics_counts_dict
        self.additional_results["sparse_metrics_names"] = sparse_metrics_names_dict
        
        # Write result to disk
        self._save_data(sparse_metrics_counts_dict, const.sparse_metrics_analysis_fpath)

    def _plot(self):
        """
        Create plot showing the number of sparse
        metrics per capital.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for iso3, df in self.output.items():
            fig, ax = plt.subplots(figsize=(7,4))
            df.plot(kind="bar", ax=ax, legend=False)
            ax.grid(True)
            ax.set_xticks(range(len(df.index)), df.index, rotation=0)
            ax.set_xlabel("Capital")
            ax.set_ylabel("Count")
            ax.set_title("Number of sparse metrics per capital")
            self._save_figure(fig, const.n_sparse_by_capital_plot_fpath)
            plt.show()


class RawDataAvailability(Analyzer):
    """
    Visualizes data availability in two different
    ways.

    Running the "analyze"-method of this class,
    a plot with two panels is generated: the
    right panel is a horizontal bar chart where
    each bar length indicates the number of
    measured data points available in the raw
    data for every metric. In the left panel,
    a heatmap shows in addition in what year
    a data point is available for any given
    metric and in what year there is no data:
    a colored tile means there is data for a
    given year and metric while a blank (white)
    tile means the corresponding data point is
    missing.

    The different colors used in the plot
    indicate to which capital (human, social,
    natural or economic) a specific metric
    belongs.

    ** IMPORTANT **
    While this class still inherits from the
    Analyzer class, the analyze method is
    overwritten to pass two different
    pandas.DataFrames to the _plot routine.

    Child of ABC "Analyzer".

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        wise_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.
        
    Attributes
    ----------
    input: pandas.DataFrame
        See input parameter "data".
        
    metrics_meta_table : pandas.DataFrame
        Table containing meta information
        for each metric.
        
    id2name_map : dict
        A dictionary mapping the metric IDs
        to the corresponding metric names.
        
    capitallist = list
        A list containing all capitals:
        - human
        - social
        - natural
        - economic
        
    overwrite : bool
        See input parameter "overwrite".
        
    output : Any
        Main result of the data analysis
        step.
        
    additional_results : dict[str, Any]
        A dictionary containing additional
        results of the data analysis step.

    Private Methods
    ---------------
    _count_datapoints_per_metric(df: pd.DataFrame) -> pd.DataFrame
    _data_availability_map(df: pd.DataFrame) -> pd.DataFrame
    _compute() -> None
    _plot() -> None

    Public Methods
    --------------
    analyze() -> None
        Is defined in parent ABC "Analyzer"
    """
    def _count_datapoints_per_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count number of available datapoints
        per metric.

        Parameters
        ----------
        df : pandas.DataFrame
            Input table considered to contain
            the measured data which is being
            counted.

        Returns
        -------
        datapoint_counts : pandas.DataFrame
            DataFrame listing all the metrics
            together with the number of datapoints
            for each metric.
        """
        datapoint_counts = df.count()\
                             .sort_values(ascending=False)\
                             .to_frame()\
                             .rename({0: "count"}, axis=1)

        datapoint_counts = datapoint_counts.join(self.metrics_meta_table)
        datapoint_counts = datapoint_counts.dropna(subset=["capital - primary"])

        return datapoint_counts
        
    def _data_availability_map(self,
                               df: pd.DataFrame
                              ) -> pd.DataFrame:
        """
        Compute data availability map.

        The map is given by a DataFrame with
        years in the columns and the metrics
        as rows (i.e. transposed w.r.t. to the
        original input data). Also, the map is
        guaranteed to have consecutive years in
        the columns and no "time gaps".

        Each entry in the resulting availability
        map is either a number (if data is
        available) or nan (it there is no data).
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input table considered to contain
            the measured data. It is this data
            of which the availability is being
            analyzed.
            
        Returns
        -------
        trp : pandas.DataFrame
            The result trp is similar to the
            input parameter df. However, the
            dataframe is now transposed, i.e.
            years are in the columns and metrics
            in the rows and there are no longer
            "time gaps", i.e. the columns are
            guaranteed to contain consecutive
            years.
        """
        # Make sure each year is represented only once
        df = df.loc[~df.index.duplicated(),:].copy()
        
        # Now transpose as we want the years to be in the
        # columns for them to be represented as a horizontal
        # time line in the heatmap later
        trp = df.transpose().copy()
        
        # Next we will fill in missing years (columns) in
        # case this is necessary. We first figure out what
        # the earliest and latest years in the data are and
        # then we make sure that every year in between is 
        # present, too, and that there are no "time gaps".
        
        # Get existing years from column index (assumed
        # datetime index)
        existing_years = [col for col in trp.columns]

        # Create a list with consecutive years between the
        # earliest and the latest year in trp.
        full_year_range = list(range(min(existing_years), 
                                     max(existing_years) + 1)
                              )
        
        # Identify missing years
        missing_years = [y for y in full_year_range
                         if y not in existing_years
                        ]
        
        # Add missing years in trp and fill
        # in nan-values
        trp = trp.reindex(columns=sorted(full_year_range))
        
        # Add capital information to each metric
        trp["capital"] = self.metrics_meta_table.loc[trp.index, "capital - primary"]

        return trp
        
    def _compute(self):
        """
        Compute the availability of measured
        data in the dataset.

        The availability is measured both in
        terms of quantity (how many measured
        data points are there per metric, i.e.
        pure count) and in terms of temporal
        availability (which metric has been
        measured in what year).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        metric_counts_dict = dict()
        metric_availability_dict = dict()
        
        for iso3, df in self.input.items():
            metric_counts = self._count_datapoints_per_metric(df)
            metric_counts_dict[iso3] = metric_counts
            metric_availability_map = self._data_availability_map(df)
    
            # Make sure both dataframes have their rows
            # sorted in the exact same way
            metric_availability_map = metric_availability_map.loc[metric_counts.index,:]
            metric_availability_dict[iso3] = metric_availability_map

        # Make results available
        self.output = metric_counts_dict
        self.additional_results["metric_availability_map"] = metric_availability_dict

        # Write result to disk
        self._save_data(metric_counts_dict, const.n_measurements_per_metrics_fpath)
        self._save_data(metric_availability_dict, const.data_availability_map_fpath)

    def _plot(self):
        """
        Creates the two-panel plot (see class
        doc-string for more details).

        Left: temporal information heatmap
        Right: count inormation bar chart

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for iso3, df in self.output.items():
            fig,axs=plt.subplots(1,2, figsize=(17,30), sharey=True, gridspec_kw = {"wspace": 0.01})

            # Left plot panel: temporal availability heatmap
            avail_df = self.additional_results["metric_availability_map"][iso3].loc[:,1850:]
            
            #Create temporal 
            axs[0] = plot.visualize_data_availability_colored(avail_df,
                                                              "Year",
                                                              "Metric Name",
                                                              "WISE metric data availability across time (since 1850)",
                                                              ax=axs[0]
                                                             )
            
            # Right plot panel: counts-only bar chart
            axs[1] = plot.raw_data_availability_barchart(df,
                                                         "Number of measured data points per metric",
                                                         "Metric Name",
                                                         "Number of measured data points per metric",
                                                         ax=axs[1],
                                                         show_legend=False
                                                         )
            
            plt.tight_layout()
            fig.suptitle(f"{iso3}: Data availability", y=0.9, fontsize=18)
            self._save_figure(fig, const.data_availability_chart_fpath, dpi=300)
            plt.show()