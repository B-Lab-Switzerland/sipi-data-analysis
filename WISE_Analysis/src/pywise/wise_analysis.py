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
        metrics_meta_table = pd.read_csv(const.wise_metatable_fname).set_index("metric_id")
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