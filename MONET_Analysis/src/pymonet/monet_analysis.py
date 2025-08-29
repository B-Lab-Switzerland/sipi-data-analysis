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
from pymonet import monet_consts as const

class Analyzer(ABC):
    """
    Abstract base class for various kinds
    of data analysis tasks in the context
    of MONET2030.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        monet_processor.py.
    
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
        capitals, and information about whether or
        not they are key metrics, we drop all other
        columns.
        The column "is_key" is aggregated as follows:
        if a metric ID is occurring multiple times,
        only one instance is retained and the "is_key"
        column of that instance is True if and only
        if at least one of all the original instances
        had an "is_key"-value equal to True.

        Parameters
        ----------
        None
        
        Returns:
        --------
        metrics_meta_table : pandas.DataFrame
            Table containing the deduplicated metric-
            level meta-information.    
        """
        metrics_meta_table = pd.read_csv(const.metrics_meta_table_fpath).set_index("metric_id")
        key_indicators = [m for m in metrics_meta_table.loc[metrics_meta_table["is_key"],:].index]

        # **REMARK**
        # Some metrics are related to more than one indicator.
        # As a result, some metrics may be repeated in the 
        # metrics_meta_table, in which case the metric-specific
        # information is identical but the related indicator-
        # or observable-level information may differ. For instance
        # a given metric can be related to two different indicators
        # and thus two different sub-SDGs, one of which is e.g. a
        # key indicator while the other is not. Both these rows
        # would still have the same row index (metric ID). This
        # duplication can lead to problems down the road, i.e. we
        # need to perform a deduplication. But for this to work
        # we can only consider metric-level information such as
        # metric ID, metric name, and capital.
        metrics_meta_table = metrics_meta_table[["metric_name", 
                                                 "metric_description",
                                                 "capital - primary"
                                                ]].drop_duplicates()

        # Now that the deduplication is done, we can add the
        # "is_key" column back in. We will set the corresponding
        # value to True if ANY of the indicators related to 
        # a given metric is a key indicator.
        metrics_meta_table["is_key"] = False
        metrics_meta_table.loc[key_indicators, "is_key"] = True

        # Sanity check
        metrics_meta_table["is_key"].sum() == len(key_indicators)
        
        return metrics_meta_table
         
    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def _plot(self, df):
        pass

    def _save_data(self, data: pd.DataFrame|dict, path: Path):
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
        if not(isinstance(data, pd.DataFrame) or isinstance(data, dict)):
            print("Data is neither a pandas.DataFrame nor a dictionary.")
            print(f"Data is of type {type(data)}.")
            return
                  
        if path.exists() and not(self.overwrite):
            print(f"File {path.as_posix()} already exists... not overwriting!")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.overwrite:
                print(f"Overwriting existing file {path.as_posix()}...")
            
            if isinstance(data, pd.DataFrame):
                assert path.as_posix().endswith(".csv")
                print(f"Writing file {path.as_posix()}...")
                data.to_csv(path)
                print("-> Done!")
                
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

    def _save_figure(self, fig: Figure, path: Path):
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
            fig.savefig(path, bbox_inches="tight")
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
        monet_processor.py.
    
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
        metrics_meta_table = self.metrics_meta_table[self.metrics_meta_table.index.str.endswith("metr")]

        kept_metrics = list(self.input.columns)
        kept_metrics_df = self.metrics_meta_table.loc[self.metrics_meta_table.index.isin(kept_metrics)]
        
        
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

        # Make results available
        self.output = n_metrics_per_cap
        
        # Write result to disk
        self._save_data(n_metrics_per_cap, const.n_metrics_per_cap_fpath)

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
        fig, ax = plt.subplots(figsize=(7,4))
        self.output.plot(kind="bar", ax=ax)
        ax.grid(True)
        ax.set_title("Number of Metrics per Capital")
        ax.set_xticks([0,1,2,3], self.output.index, rotation=0)
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
        monet_processor.py.
    
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

        sparse_metrics = list(self.input["sparse columns (<10 data points)"].values)
        sparse_metrics_df = self.metrics_meta_table.loc[self.metrics_meta_table.index.isin(sparse_metrics)]
        sparse_metrics_names = sparse_metrics_df.groupby("capital - primary")\
                                                .agg({"metric_name": lambda x: ", ".join(x)})
        sparse_metrics_df = sparse_metrics_df.groupby("capital - primary")\
                                             .agg({"metric_name": "count"})\
                                             .rename({"metric_name": "count"}, axis=1)\
                                             .sort_values(by="count", ascending=False)

        # Make results available
        self.output = sparse_metrics_df
        self.additional_results["sparse_metrics_names"] = sparse_metrics_names
        
        # Write result to disk
        self._save_data(sparse_metrics_df, const.sparse_metrics_analysis_fpath)

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
        fig, ax = plt.subplots(figsize=(7,4))
        self.output.plot(kind="bar", ax=ax, legend=False)
        ax.grid(True)
        ax.set_xlabel("Capital")
        ax.set_ylabel("Count")
        ax.set_title("Number of sparse metrics per capital")
        self._save_figure(fig, const.n_sparse_by_capital_plot_fpath)
        plt.show()


class nIrrelevantMetricsPerCapital(Analyzer):
    """
    Analyze number of metrics per capital considered
    irrelevant for agenda2030.

    Child of ABC "Analyzer".

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        monet_processor.py.
    
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
        Count the number of irrelevant metrics per 
        capital.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        irrelevant_metrics = list(self.input.columns)
        irrelevant_metrics_df = self.metrics_meta_table.loc[self.metrics_meta_table.index.isin(irrelevant_metrics)]
        irrelevant_metrics_names = irrelevant_metrics_df.groupby("capital - primary")\
                                                        .agg({"metric_name": lambda x: ", ".join(x)})
        irrelevant_metrics_counts = irrelevant_metrics_df.groupby("capital - primary")\
                                                         .agg({"metric_name": "count"})\
                                                         .rename({"metric_name": "count"}, axis=1)\
                                                         .sort_values(by="count", ascending=False)

        # Make results available
        self.output = irrelevant_metrics_counts
        self.additional_results["irrelevant_metrics_names"] = irrelevant_metrics_names
        
        # Write result to disk
        self._save_data(irrelevant_metrics_counts, const.irrelevant_metrics_analysis_fpath)

    def _plot(self):
        """
        Create plot showing the number of irrelevant
        metrics per capital.

        Parameters
        ----------
        df: pandas.DataFrame
            Table containing the counts

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize=(7,4))
        self.output.plot(kind="bar", ax=ax, legend=False)
        ax.grid(True)
        ax.set_xlabel("capital")
        ax.set_ylabel("count")
        ax.set_title("Number of metrics irrelevant to agenda2030 (per capital)")
        self._save_figure(fig, const.n_irrelevant_by_capital_plot_fpath)
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

    Child of ABC "Analyzer".

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        monet_processor.py.
    
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
        trp = df.transpose()
        
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
        
        # Add missing columns to the right of trp and fill
        # in nan-values
        for year in missing_years:
            trp[year] = np.nan * len(trp)
        
        # Reorder columns chronologically
        trp = trp.reindex(sorted(trp.columns), axis=1)

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
        metric_counts = self._count_datapoints_per_metric(self.input)
        metric_availability_map = self._data_availability_map(self.input)

        # Make sure both dataframes have their rows
        # sorted in the exact same way
        metric_availability_map = metric_availability_map.loc[metric_counts.index,:]

        # Make results available
        self.output = metric_counts
        self.additional_results["metric_availability_map"] = metric_availability_map

        # Write result to disk
        self._save_data(metric_counts, const.n_measurements_per_metrics_fpath)
        self._save_data(metric_availability_map, const.data_availability_map_fpath)

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
        fig,axs=plt.subplots(1,2, figsize=(17,30), sharey=True, gridspec_kw = {"wspace": 0.01})
        
        # Left plot panel: temporal availability heatmap
        axs[0] = plot.visualize_data_availability_colored(self.additional_results["metric_availability_map"],
                                                          "Year",
                                                          "Metric Name",
                                                          "MONET2030 metric data availability across time",
                                                          ax=axs[0]
                                                         )
        
        # Right plot panel: counts-only bar chart
        axs[1] = plot.raw_data_availability_barchart(self.output,
                                                     "Number of measured data points per metric",
                                                     "Metric Name",
                                                     "Number of measured data points per metric",
                                                     ax=axs[1],
                                                     show_legend=False
                                                     )
        
        plt.tight_layout()
        fig.suptitle("Data availability", y=0.9, fontsize=18)
        self._save_figure(fig, const.data_availability_chart_fpath)
        plt.show()


class CorrleationAnalysis(Analyzer):
    """
    Perform all analysis steps related to 
    cross-correlating the different metrics.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        monet_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.

    Optional Parameters
    -------------------
    lag: int [default: 0]
        Lag at which the cross-correlations
        shall be computed. When lag<0, then
        the cross-correlations are computed
        at all possible lags and max/min-
        aggregated. See documentation of
        utils.CorrelationAnalysis for more
        details.
    
    threshold_vector: Iterable|None [default: None]
        Vector of thresholds at which the
        metric pruning shall be performed.
        When is set to None, it will be
        automatically set to
        [th/100 for th in range(80,100,2)]\
        +[0.99, 0.999]
        
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

    lag : int
        See input parameter "lag".
    
    infix = f"lag{self.lag}" if self.lag>=0 else f"aggregated"
        Infix used to modify path names.
        
    th_vec : List[float]
        See input parameter "threshold_vector"

    Private Methods
    ---------------
    _filter_metrics_by_correlation(corranalysis_obj: utils.CorrelationAnalysis,
                                   corrmat: pandas.DataFrame
                                  ) -> Tuple[pandas.DataFrame,
                                             dict[float, List[str]],
                                             dict[float, dict[str, pandas.DataFrame]]
                                            ]
    _plot_number_of_redundant_metrics() -> None
    _plot_corrval_distributions() -> None
    _compute() -> None
    _plot() -> None
    
    Public Methods
    --------------
    analyze() -> None
        Is defined in parent ABC "Analyzer"
    """
    def __init__(self,
                 data: pd.DataFrame,
                 lag: int = 0,
                 threshold_vector: Iterable|None = None,
                 key_indicators_only: bool = False
                ):
        super().__init__(data)
        self.key_indicators_only = key_indicators_only
        if self.key_indicators_only:
            key_metric_names = list(self.metrics_meta_table.loc[self.metrics_meta_table["is_key"],:].index)
            self.input = self.input[[c for c in self.input.columns if c in key_metric_names]]

        # Additional parameters
        self.lag = lag
        self.infix = f"lag{self.lag}" if self.lag>=0 else f"aggregated"
        if self.key_indicators_only:
            self.infix += "_key_indicators_only"
        self.th_vec = threshold_vector
        if self.th_vec is None:
            self.th_vec = [th/100 for th in range(80,100,2)]+[0.99, 0.999]

        # Containers
        self.additional_results = dict()

    def _filter_metrics_by_correlation(self,
                                       corranalysis_obj: utils.CorrelationAnalysis,
                                       corrmat: pd.DataFrame
                                      ) -> Tuple[pd.DataFrame,
                                                 dict[float, List[str]],
                                                 dict[float, dict[str, pd.DataFrame]]
                                                ]:
        """
        Peform metric pruning based on correlation
        analysis.

        For full information about how the pruning
        is done, see documentation of 
        utils.CorrelationAnalysis.

        The pruning is done for each value in the
        vector in self.th_vec.

        Parameters
        ----------
        corranalysis_obj : utils.CorrelationAnalysis
            CorrelationAnalysis object containing
            required correlation analysis data.
            
        corrmat : pandas.DataFrame
            Dataframe containing all pairwise 
            correlations for each pair of metrics.
            
        Returns
        -------
        counts : pandas.DataFrame
            DataFrame containing number
            of retained metrics after pruning
            for each value in self.th_vec (columns)
            per capital (rows).
            
        to_keep_dict : dict[float, List[str]]
            Lists of metrics retained after
            pruning at threshold value th.
            The threshold values are the keys
            of the dictionary, the lists are
            the values.
        
        corr_groups : dict[float, dict[str, pandas.DataFrame]]
            A dictionary containing key-value pairs
            where the key is a threshold value th
            and the value is again a dictionary. The
            inner dictionary contains a metric name
            as the keys and the corresponding 
            values (pandas.DataFrames) list all
            the metrics which are correlated more
            strongly than the threshold value th
            with the metric in the key.
        """
        # Initialize containers
        counts_list = []
        to_keep_dict = dict()
        corr_groups = dict()

        # Iterate over all thresholds
        for th in self.th_vec:
            to_keep, corr_xlsx = corranalysis_obj.drop_strong_correlations(corrmat, 
                                                                           threshold=th 
                                                                           )

            # Count number of kept metrics per capital
            to_keep_meta_df = self.metrics_meta_table[self.metrics_meta_table.index.isin(to_keep)]
            counts_per_cap = to_keep_meta_df.groupby("capital - primary")\
                                            .agg("count")\
                                            .rename({"metric_name": th}, axis=1)

            # Notice that at this point the variables
            # to_keep and corr_xlsx contain only metric_ids.
            # This is not userfriendly. Therefore, we now
            # need to translate those ids into metric
            # names for user to be able to intuitively
            # understand what they are looking at.
            #to_keep_names = list(to_keep_meta_df["metric_name"].values)
            to_keep_names = to_keep_meta_df[["metric_name", "is_key"]]
            corr_group_names = dict()
            for mid, ser in corr_xlsx.items():
                m_name = self.id2name_map[mid]
                
                named_corrgroup = ser.to_frame()\
                                     .join(self.metrics_meta_table, 
                                           how="inner"
                                          )
                named_corrgroup = named_corrgroup[["metric_name", mid]]
                named_corrgroup = named_corrgroup.rename({mid: "correlation"}, axis=1)
                corr_group_names[m_name] = named_corrgroup
            
            # Fill results into containers
            counts_list.append(counts_per_cap[th])
            to_keep_dict[th] = to_keep_names
            corr_groups[th] = corr_group_names#corr_xlsx

        # Create a single counts dataframe
        counts = pd.concat(counts_list, axis=1)
        counts.loc["Total", :] = counts.sum(axis=0) 

        # Return
        return counts, to_keep_dict, corr_groups

    def _count_retained_key_metrics_after_pruning(self, to_keep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Count how many metrics retained after pruning at
        the various different correlation threshold 
        are associated with key-indicators. In addition
        to the absolute counts, also compute the fraction
        among all the retained metrics.

        Parameters
        ----------
        to_keep_df : pandas.DataFrame
            DataFrame containing the retained metrics after
            pruning and the corresponding "is_key" flags.

        Returns
        -------
        retained_counts_df : pandas.DataFrame
            A table containing the absolute counts (column 1)
            and the fraction among all metrics (column 2)
            for each correlation threshold th (rows).
        """
        counts = [to_keep_df[th]["is_key"].sum() for th in self.th_vec]
        fracts = [to_keep_df[th]["is_key"].sum()/to_keep_df[th]["is_key"].count() for th in self.th_vec]
        retained_counts_df = pd.DataFrame({"counts": counts, "fractions": fracts}, index=self.th_vec)
        return retained_counts_df
        
    def _compute(self):
        """
        Run the correlation analysis and use the
        results to prune the list of MONET2030
        metrics using several different correlation
        threshold values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        monet_ca = utils.CorrelationAnalysis(self.input)

        if self.lag >= 0:
            corrmat = monet_ca.cross_corr(lag=self.lag)
        else:
            corrmat = monet_ca.max_abs_corr()

        counts, to_keep, corr_groups = self._filter_metrics_by_correlation(monet_ca, corrmat)
        retained_counts_df = self._count_retained_key_metrics_after_pruning(to_keep)

        # Make results available
        self.output = counts
        self.additional_results["correlation_matrix"] = corrmat
        self.additional_results["metrics_to_keep"] = to_keep
        self.additional_results["correlation_groups"] = corr_groups
        self.additional_results["retained_key_after_pruning"] = retained_counts_df 

        # Write result to disk
        self._save_data(corrmat, const.all_corrmat_fpath(self.infix))
        self._save_data(counts, const.metric_counts_fpath(self.infix))
        self._save_data(to_keep, const.to_keep_fpath(self.infix))
        self._save_data(retained_counts_df, const.n_retained_key_metrics_fpath(self.infix))
        for th, group in corr_groups.items():
            thstr = str(th).replace(".","p")
            fpath = const.corr_groups_fpath(self.infix,thstr)
            
            if group == {}:
                print(f"File {fpath} would be empty --> not writing...")
                continue
            
            self._save_data(group, fpath)

    def _plot_number_of_redundant_metrics(self):
        """
        Plot the number of metrics retained
        after pruning vs a list of different
        correlation thresholds.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        counts_df = self.output
        
        fig, ax = plt.subplots(figsize=(10,4))
        counts_df.T.plot(kind="line", ax=ax)
        ax.set_xticks(counts_df.columns, counts_df.columns, rotation=60)

        maxval = int(counts_df.max().max())
        yticks = range(0, maxval+1, 2)
        if len(yticks)>20:
            yticks = range(0, maxval+4, 5)
        if len(yticks)>20:
            yticks = range(0, maxval+9, 10)
            
        ax.set_yticks(yticks, yticks)
        ax.set_xlabel("correlation threshold")
        ax.set_ylabel("Number of non-redundant metrics")
        ax.set_title("Counting non-redundant metrics in dependency of\ncorrelation threshold")
        if self.key_indicators_only:
            ax.set_title("Counting non-redundant metrics in dependency of correlation threshold\n(considering only key indicator-associated metrics)")
        ax.grid()
        plt.tight_layout()
        self._save_figure(fig, const.metric_counts_plot_fpath(self.infix))
        plt.show()

    def _plot_corrval_distributions(self):
        """
        Plot histogram of correlation values.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        ax = axs[0]
        sns.histplot(self.additional_results["correlation_matrix"].unstack().values, 
                     ax=ax, 
                     kde=True, 
                     bins=40)
        ax.set_title("Distribution of correlation values")
        ax.set_xlabel("corr")
        
        ax = axs[1]
        sns.histplot(self.additional_results["correlation_matrix"].unstack().abs().values,
                     ax=ax,
                     bins=20)
        ax.set_title("Distribution of abs(correlation) values")
        if self.key_indicators_only:
            ax.set_title("Distribution of abs(correlation) values\n(considering key indicator-associated metrics only)")
        ax.set_xlabel("abs(corr)")
        plt.tight_layout()
        self._save_figure(fig, const.corr_val_distribution_plot_fpath(self.infix))
        plt.show()

    def _plot_n_key_metrics_retained(self):
        """
        Create a plot showing how many of the
        retained metrics (after pruning at
        correlation threshold th) are associated
        with key indicators.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        th_vec = self.th_vec
        xvec = range(len(th_vec))
        counts = self.additional_results["retained_key_after_pruning"]["counts"]
        fracts = self.additional_results["retained_key_after_pruning"]["fractions"]
        
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(xvec, counts, marker="d", c="blue")
        axs[1].plot(xvec, fracts, marker="d", c="orange")
        axs[1].set_xticks(xvec, th_vec)
        axs[1].set_xlabel("correlation threshold")
        axs[0].set_ylabel("absolute count")
        axs[1].set_ylabel("fraction")
        axs[0].set_title("Absolute number and fractions of key indicator-associated metrics\nretained after pruning at various correlation thresholds")
        axs[0].grid(True)
        axs[1].grid(True)
        self._save_figure(fig, const.n_retained_key_metrics_plot_fpath(self.infix))
        plt.show()
        
    def _plot(self):
        """
        Create the cross-correlation and pruning
        visualizations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._plot_number_of_redundant_metrics()
        self._plot_corrval_distributions()
        if not self.key_indicators_only:
            self._plot_n_key_metrics_retained()

class PerformanceRanker(Analyzer):
    """
    Parameters
    ----------
    data: pandas.DataFrame
        Input data being analyzed. This is
        meant to be the result of a data
        transformation step defined in 
        monet_processor.py.
    
    overwrite : bool
        A flag indicating whether or not
        results should be saved to files
        overwriting already existing files.

    Optional Parameters
    -------------------
    key_indicators_only: bool [default: False]
        Flag indicating whether or not the
        analysis should be done only using
        metrics associated to MONET2030
        key-indicators.
        
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

    trend_targets : pandas.DataFrame
        Table containing the desired trend
        for each metric as published on the
        WWW.

    key_indicators_only : bool
        See input argument "key_indicators_only".

    Private Methods
    ---------------
    _get_desired_trends
    _plot_number_of_redundant_metrics
    _plot_corrval_distributions(corrmat)
    _compute()
    _plot()
    
    Public Methods
    --------------
    analyze() -> None
        Is defined in parent ABC "Analyzer"
    """
    def __init__(self,
                 data: pd.DataFrame,
                 key_indicators_only: bool=False
                ):
        super().__init__(data)
        self.trend_targets = self._get_desired_trends()
        self.key_indicators_only = key_indicators_only
        
    def _get_desired_trends(self) -> pd.DataFrame:
        """
        Read in the desired/targeted directions from
        a file on disk. This informatino is scraped
        from the WWW.

        Returns
        -------
        directions_with_names : pandas.DataFrame
            Table containing the desired trend
            for each metric as published on the
            WWW.
        """
        directions = pd.read_csv(const.trend_directions).set_index("metric_id")
        directions["dam_id"] = [int(metr_id.split("_")[0][:-1]) for metr_id in directions.index]
        directions_with_names = directions.join(self.metrics_meta_table[["metric_name"]])
        directions_with_names = directions_with_names[["metric_name", 
                                                       "observable", 
                                                       "description", 
                                                       "dam_id", 
                                                       "desired_trend"]]
        return directions_with_names

    def _compute_slope(self, mid: str) -> Tuple[float, float]:
        """
        Computes the slope and the normalized
        slope for the metric specified by
        the metric ID "mid".

        The normalized slope is equal to the
        slope up to the sign, which depends
        on the desired trend. If the desired
        trend of a given metric is "down", then
        the normlalized slope is (-1) times
        the slope. As a result, a higher
        normalized slope is always better
        (while for the slope it depends on the
        desired trend whether a higher or a 
        lower value is better).

        Parameters
        ----------
        mid : str
            a metric ID

        Returns
        -------
        slope : float
            The actual slope of the linear
            interpolation of the values
            for the metric mid.

        slope_norm : float
            Normalized slope for the metric
            mid as defined above.
        """
        data = self.input[mid].dropna()
        x = [year for year in data.keys()]
        y_norm = data.values/data.values[0]
        slope = np.polyfit(x,y_norm,1)[0]
            
        if self.trend_targets.loc[mid, "desired_trend"]=="up":
            slope_norm = slope
        elif self.trend_targets.loc[mid, "desired_trend"]=="down":
            slope_norm = -slope
        else:
            slope_norm = np.nan

        return slope, slope_norm

    def _best3_metrics_per_capital(self, 
                                   rankings: pd.DataFrame, 
                                   cap: str|None=None
                                  ) -> pd.DataFrame:
        """
        Extract three top-performing metrics.

        The ranking is done by normalized slope.
        The higher the normalized slope the better.
        Therefore, this method returns the three
        metrics with highest normalized slopes
        (either overall or for a specified capital).

        Parameters
        ----------
        rankings : pandas.DataFrame
            DataFrame containing the MONET2030
            metrics ranked by their normalized
            slopes from highest to lowest.

        Optional Parameters
        -------------------
        cap : str [default: None]
            A specific capital (i.e. "Social", 
            "Human", "Natural", or "Economic").
            If not None, the three top metrics
            within the defined capital are returned.
            Otherwise, the capital aspect is 
            ignored and the top 3 metrics over 
            all capitals are returned.

        Returns
        -------
        top3 : pandas.DataFrame
            A dataframe containing the three
            top-performing metrics.
        """
        if cap is None:
            top3 = rankings.head(3)
        else:
            top3 = rankings[rankings["capital - primary"]==cap].head(3)

        return top3

    def _worst3_metrics_per_capital(self, 
                                    rankings: pd.DataFrame, 
                                    cap: str|None=None
                                   ) -> pd.DataFrame:
        """
        Extract three worst-performing metrics.

        The ranking is done by normalized slope.
        The higher the normalized slope the better.
        Therefore, this method returns the three
        metrics with lowest normalized slopes
        (either overall or for a specified capital).

        Parameters
        ----------
        rankings : pandas.DataFrame
            DataFrame containing the MONET2030
            metrics ranked by their normalized
            slopes from highest to lowest.

        Optional Parameters
        -------------------
        cap : str [default: None]
            A specific capital (i.e. "Social", 
            "Human", "Natural", or "Economic").
            If not None, the three worst metrics
            within the defined capital are returned.
            Otherwise, the capital aspect is 
            ignored and the three worst metrics
            over all capitals are returned.

        Returns
        -------
        bottom3 : pandas.DataFrame
            A dataframe containing the three
            worst-performing metrics.
        """
        if cap is None:
            bottom3 = rankings.tail(3)
        else:
            bottom3 = rankings[rankings["capital - primary"]==cap].tail(3)

        return bottom3

    def rank_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the ranking of the MONET2030
        metrics based on their normalized
        slope. See documentation of 
        "_compute_slope" for more information.

        Parameters
        ----------
        None

        Returns
        -------
        ranking : pandas.DataFrame
            DataFrame containing the MONET2030
            metrics ranked by their normalized
            slope (from highest to lowest).

        slope_norm_stats : pandas.DataFrame
            DataFrame containing summary statistics
            of the normalizes slopes per capital.
        """
        # Get a table with all metrics including their name and desired_trend and 
        # prepare two columns for slope information: slope and slope_norm        
        ranking = self.trend_targets.copy()
        ranking[["slope", "slope_norm"]] = None

        # Fill in the slope and slope_norm columns

        # REMARK: slope_norm is introduced the put all metrics
        # on a common scale. Depending on the desired_trend, for
        # some metrics a higher (i.e. more positive) slope is 
        # better, for other metrics a lower (i.e. more negative)
        # slope is better. The column slope_norm simplifies this
        # as a higher (more positive) slope_norm is always better.
        
        for mid in ranking.index:
            slope, slope_norm = self._compute_slope(mid)
            
            ranking.loc[mid, "slope"] = slope
            ranking.loc[mid, "slope_norm"] = slope_norm
                            
        ranking = ranking.join(self.metrics_meta_table[["capital - primary", "is_key"]])
        ranking = ranking.dropna(subset="slope_norm").sort_values(by="slope_norm", ascending=False)

        # Get summary statistics about the slope_norm
        slope_norm_stats = ranking.groupby("capital - primary").agg({"slope_norm": ["mean", "median", "std"]})
        
        if self.key_indicators_only:
            ranking = ranking[ranking["is_key"]]
        else:
            ranking = ranking
            
        return ranking, slope_norm_stats

    def get_top3(self, ranking: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute the three top-performing metrics
        for each capital and overall. Also, combine
        the top-performing metrics of per capital
        into a combined view.

        Parameters
        ----------
        ranking : pandas.DataFrame
            DataFrame containing the MONET2030
            metrics ranked by their normalized
            slope (from highest to lowest).

        Returns
        -------
        top3 : Dict[str, pd.DataFrame]
            A dict containing the three top-performing
            metrics for each of the four capitals as
            well as over all the capitals.
        """
        if self.key_indicators_only:
            ranking = ranking[ranking["is_key"]]
            
        # Best performers
        top3 = dict()
        top3["overall"] = self._best3_metrics_per_capital(ranking)
        for cap in self.capitallist:
            top3[cap] = self._best3_metrics_per_capital(ranking, cap=cap)

        top3["combined"] = pd.concat([top3[cap] for cap in self.capitallist])

        return top3 
            
    def get_bottom3(self, ranking: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute the three worst-performing metrics
        for each capital and overall. Also, combine
        the top-performing metrics of per capital
        into a combined view.

        Parameters
        ----------
        ranking : pandas.DataFrame
            DataFrame containing the MONET2030
            metrics ranked by their normalized
            slope (from highest to lowest).

        Returns
        -------
        top3 : Dict[str, pd.DataFrame]
            A dict containing the three worst-performing
            metrics for each of the four capitals as
            well as over all the capitals.
        """
        if self.key_indicators_only:
            ranking = ranking[ranking["is_key"]]
        # Worst performers
        bottom3 = dict()
        bottom3["overall"] = self._worst3_metrics_per_capital(ranking)
        for cap in self.capitallist:
            bottom3[cap] = self._worst3_metrics_per_capital(ranking, cap=cap)

        bottom3["combined"] = pd.concat([bottom3[cap] for cap in self.capitallist])

        return bottom3

    def _compute(self):
        """
        Compute the ranking of all MONET2030 metrics
        based on their normalized slope and extract
        the top and worst performing metrics based
        on this ranking. Also, extract some summary
        statistics about the normalized slopes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ranking, slope_stats = self.rank_metrics()

        if self.key_indicators_only:
            ranking = ranking[ranking["is_key"]]
        best = self.get_top3(ranking)
        worst = self.get_bottom3(ranking)

        # Make results available
        self.output = ranking
        self.additional_results["normalized_slope_statistics"] = slope_stats
        self.additional_results["top_performers"] = best
        self.additional_results["bottom_performers"] = worst

        # Write result to disk
        if self.key_indicators_only:
            self._save_data(ranking, const.key_indicator_ranking_fpath)
            self._save_data(slope_stats, const.key_indicator_slope_stats_fpath)
            self._save_data(best, const.key_indicator_top_performers_fpath)
            self._save_data(worst, const.key_indicator_bottom_performers_fpath)
        else:
            self._save_data(ranking, const.ranking_fpath)
            self._save_data(slope_stats, const.slope_stats_fpath)
            self._save_data(best, const.top_performers_fpath)
            self._save_data(worst, const.bottom_performers_fpath)

    def _plot_ranking_distributions(self):
        """
        Create box plot showing distribution of
        normalized slopes per capital.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize=(10,3))
        sns.boxplot(data=self.output, x="capital - primary", y="slope_norm", ax=ax)
        ax.grid(True)
        if self.key_indicators_only:
            ax.set_title("Distribution of normalized slopes per capital (key indicators only)")
            ax.set_ylabel("normalized slope distribution")
        else:
            ax.scatter([2],[-0.045], marker="v", color="red", label="direction of single extreme outlier")
            ax.set_ylim([-0.05,0.3])
            ax.legend(loc="upper right")
            ax.set_title("Distribution of normalized slopes per capital")
        if self.key_indicators_only:
            self._save_figure(fig, const.key_indicator_slope_distro_plot_fpath)
        else:
            self._save_figure(fig, const.slope_distro_plot_fpath)
        plt.show()

    def _plot_metrics_performance_ranking(self):
        """
        Create horizontal bar chart showing the
        normalized slopes for each metric in 
        decreasing order.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        df = self.output
        if self.key_indicators_only: 
            fig, ax = plt.subplots(figsize=(15,6))
        else:
            fig, ax = plt.subplots(figsize=(15,30))
        # Reset index for plotting
        df_plot = df.reset_index()
        # Use a numeric x axis to avoid categorical spacing issues
        df_plot["y_pos"] = range(len(df_plot))
            
        # Plot bars
        sns.barplot(
            data=df_plot,
            x="slope_norm", y="y_pos", hue="capital - primary",
            dodge=False,  # Keep all hues at the same x-position
            orient="horizontal",
            ax=ax,
            legend=True
        )
        
        # Optional: show fewer x-ticks
        yticks = df_plot["y_pos"]
        ylabels = df_plot["metric_name"]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        
        # Labels and title
        ax.set_xlabel("normalized slope (higher = better)")
        if self.key_indicators_only:
            ax.set_ylabel("key metric names")
            ax.set_title("Ranking of evolution over time of MONET2030 metrics (key indicators only)")
        else:
            ax.set_ylabel("metric names")
            ax.set_title("Ranking of evolution over time of MONET2030 metrics")
        ax.grid(True)
        
        plt.tight_layout()
        if self.key_indicators_only:
            self._save_figure(fig, const.key_indicator_ranking_plot_fpath)
        else:
            self._save_figure(fig, const.ranking_plot_fpath)
        plt.show()

    def _plot_nkey_per_group(self) -> None:
        """
        Creates a stacked bar chart showing
        what fraction of all the best- and
        worst performing metrics is in the
        group of key indicators.
    
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """
        n_key_among_top = self.additional_results["top_performers"]["combined"]["is_key"].sum()
        n_key_among_bottom = self.additional_results["bottom_performers"]["combined"]["is_key"].sum()
        
        fig, ax = plt.subplots()
        ax.bar([0,1], 
               height=[n_key_among_top, n_key_among_bottom],
               facecolor="blue",
               label="key indicators"
              )
        ax.bar([0,1],
               height=[12-n_key_among_top, 12-n_key_among_bottom],
               bottom=[n_key_among_top, n_key_among_bottom],
               facecolor="orange",
               label="non-key indicators"
              )
        ax.grid(True)
        ax.set_xticks([0,1], ["top 3 for all capitals", "bottom 3 for all capital"])
        ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.25], ncols=2)
        ax.set_xlabel("Performance group")
        ax.set_ylabel("Counts")
        ax.set_title("Fraction of key indicator metrics per performance group")
        fig.savefig(const.n_key_indicators_per_performance_plot_fpath, bbox_inches="tight")
        plt.show()

    def _plot(self):
        """
        Create the ranking plot (horizontal
        bar chart) and the box plot for the
        normalized statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._plot_ranking_distributions()
        self._plot_metrics_performance_ranking() 
        if not self.key_indicators_only:
            self._plot_nkey_per_group()

        
