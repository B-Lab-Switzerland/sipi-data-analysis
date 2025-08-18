# Std lib imports
import os
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime as dt
from typing import Dict, List, Tuple, Any
from pathlib import Path

# 3rd party imports
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from sklearn.preprocessing import StandardScaler

# Local imports
from pywise import wise_aux as aux
from pywise import wise_consts as const
#from pywise import wise_logger as logger
from sipi_da_utils import clean, impute, tsa, plot

class Processor(ABC):
    """
    Abstract base class from which each data
    transformation step for the WISE data
    analysis will be subclassed.

    A series of subclasses of the 'Processor' 
    ABC can be used to transform the WISE
    indicator database downloaded in the form of
    ill-formatted excel spreadsheets from the www
    into nicely structured dataframes (which
    eventually are processed at different levels,
    including data cleaning, data imputation etc.)

    Attributes
    ----------
    input : Any
        Input data to be transformed.
        The format is different depending
        on the data processing stage.

    metatable : pandas.DataFrame
        DataFrame containing additional
        metainformation about each WISE
        indicator.

    stage_name : int
        Enumerates the current stage.
        Starts counting from 1.
    
    Abstract methods
    ----------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool:

    Concrete methods
    ----------------
    get_data(force: bool) -> None
    """
    def __init__(self, 
                 input_data: pd.DataFrame,
                 metatable: pd.DataFrame,
                 stage_id: int,
                 verbosity: int,
                 ):

        # Input data
        self.input = input_data
        self.metatable = metatable
        self.stage_id = stage_id

        # Further variable declarations
        self.output: Any = None  # This will ultimately store the final result
        self.additional_results: Dict[str, Any] = dict()
        self.previous_stage_fpath: Path = const.processed_dir / f"stage_{stage_id-1}" if stage_id >= 1  else const.raw_dir
        self.current_stage_fpath: Path = const.processed_dir / f"stage_{stage_id}"
        self.log: Dict = dict()
        self.verbosity = verbosity

    @abstractmethod
    def _transform(self):
        pass
    
    @abstractmethod
    def _read(self):
        pass

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _is_done(self) -> bool:
        pass

    def get_data(self, force: bool=False) -> Any:
        """
        Main method to transform data.
        
        Checks whether to read data from disk or 
        to compute it from previously executed
        transformation steps.

        Parameters
        ----------
        force : bool [default: False]
            If True, forces recreation/recomputation
            from previous data transformation stage
            even if current stage data is available
            on disk.

        Returns
        -------
        output : Any
            Data after application of current-stage
            transformation logic.
        """
        # Read data from disk if it already exists
        
        if (not force) & self._is_done():
            self._read()
    
        # Otherwise, transform the raw data
        else:
            self._transform()

        return self.output

class Stage1(Processor):
    """
    Reformats the data from a single long-format
    table into a set of wide-format pivot tables
    with the columns referring to the individual
    metrics and the rows referring to individual
    years.

    The result of this transformation step is a
    dictionary of pandas.DataFrames that can be
    saved to an Excel spreadsheet. Each key-value
    pair or work sheet, respectively, contains
    the data of a single country as specified
    by the key/work sheet name.

    Class attributes
    ----------------
    c_name : str
        class name = "stage1"

    c_description : str
        ultra-short class description = "standardize data"

    Private Methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    
    Public Methods
    --------------
    get_data(force: bool) -> None
        Is defined in parent ABC "Processor"
    """
    c_name = "stage1"
    c_description = "standardize data"

    def _transform(self):
        """
        """
        raw_indices = self.input["C Data"]
        country_vec = list(raw_indices["ISO3"].values)

        metric_tables_dict = dict()
        for iso3 in country_vec:
            country_table = raw_indices[raw_indices["ISO3"] == iso3].pivot(index="Acronym", 
                                                                           columns="Year", 
                                                                           values="Value"
                                                                          )
            country_table = country_table.transpose()
            country_table.index = country_table.index.astype(int)
            country_table.index.name = "year"
            country_table.columns.name = "metric_id"
            country_table = aux.standardize_column_names(country_table)
            metric_tables_dict[iso3] = country_table
            
        self.output = metric_tables_dict
        self._save(const.wise_metric_tables_fname, self.output)

        print("-> done!")

    def _read(self):
        """
        Read clean data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading clean data from disk...")

        self.output = pd.read_excel(self.current_stage_fpath / const.wise_metric_tables_fname, 
                                    sheet_name = None, 
                                    index_col="year"
                                   )
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if stage 1 data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        metrics_table_file_exists = (self.current_stage_fpath / const.wise_metric_tables_fname).exists()
        
        is_done = metrics_table_file_exists
        return is_done

    def _save(self, fname: str, datadict: dict):
        """
        Writes cleaned data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the WISE time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dirpath / fname) as writer:
            for name, df in datadict.items():
                df.to_excel(writer, sheet_name=name)


class WiseDataCleaning(Processor):
    """
    Data cleaning functionality.

    Run through a set of different data cleaning
    steps as explained in more detail in the
    docstring of the _transform method.
    
    Class attributes
    ----------------
    c_name : str
        class name = "Data cleaner"

    c_description : str
        ultra-short class description = "Perform generic and dataset-specific cleaning steps"
        
    Private methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """
    c_name = "Data cleaner"
    c_description = "Perform generic and dataset-specific cleaning steps"
        
    def _transform(self):
        """
        Clean the data.

        First, a dataset-specific cleaning step is performed
        in which only the metrics (columns) are retained that
        are agenda2030-relevant.

        Next, a series of generic data cleaning steps is
        performed:
        - de-duplication of rows
        - removal of columns that have constant values
        - application of a time window filter keeping only rows
          corresponding to years inside that time window
        - removal of columns that do not contain a minimum
          number of data points (sparse columns)
        """
        # Perform generic cleaning steps
        year_min = 1900
        year_max = 2025
        min_data_points = 10

        cleaner_dict = dict()
        duplicated_rows = dict()
        constant_cols = dict()
        outside_years = dict()
        sparse_cols = dict()
        for sheet, df in self.input.items():
            assert df.index.name == "year"
        
            cleaner = clean.DataCleaner(df, verbose=self.verbosity)
            duplicated_rows[sheet] = cleaner.drop_duplicates()
            constant_cols[sheet] = pd.Series(cleaner.remove_constant_columns(), 
                                             name="constant columns"
                                            )
            outside_years[sheet] = pd.Series(cleaner.apply_time_filter(min_year = year_min, max_year = year_max),
                                             name=f"years outside [{year_min}, {year_max}]"
                                            )
            sparse_cols[sheet] = pd.Series(cleaner.drop_sparse_columns(n_notnull_min = min_data_points),
                                           name=f"sparse columns (<{min_data_points} data points)"
                                          )
            cleaner_dict[sheet] = cleaner

        # Make data available
        cleaner.df.index.name = "year"
        self.output = {sheet: cleaner.df for sheet, cleaner in cleaner_dict.items()}
        self.additional_results["duplicated_rows"] = duplicated_rows
        self.additional_results["constant_cols"] = constant_cols
        self.additional_results["outside_years"] = outside_years 
        self.additional_results["sparse_cols"] = sparse_cols
        # Write processed data to csv files
        self._save(const.clean_data_fname, self.output)
        self._save(const.duplicated_rows_fname, self.additional_results["duplicated_rows"])
        self._save(const.constant_cols_fname, self.additional_results["constant_cols"])
        self._save(const.sparse_cols_fname, self.additional_results["sparse_cols"])
        
        [root, extension] = const.outside_years_fname.split(".")
        self._save(f"{root}_{year_min}to{year_max}.{extension}", self.additional_results["outside_years"])

        print("-> done!")
        
        
    def _read(self):
        """
        Read clean data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading clean data from disk...")

        self.output = pd.read_excel(self.current_stage_fpath / const.clean_data_fname, 
                                    sheet_name=None,
                                    index_col="year"
                                   )

        self.additional_results["duplicated_rows"] = pd.read_excel(self.current_stage_fpath / const.duplicated_rows_fname, 
                                                                   sheet_name=None,
                                                                   index_col="year"
                                                                  )
        self.additional_results["constant_cols"] = pd.read_excel(self.current_stage_fpath / const.constant_cols_fname,
                                                                 sheet_name=None
                                                                )
        self.additional_results["sparse_cols"] = pd.read_excel(self.current_stage_fpath / const.sparse_cols_fname,
                                                               sheet_name=None
                                                              )

        [root, extension] = const.outside_years_fname.split(".")
        outside_years_fname = [f for f in self.current_stage_fpath.glob('**') if root in f.as_posix()][0]
        self.additional_results["outside_years"] = pd.read_excel(outside_years_fname, sheet_name=None)
        
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if stage 1 data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.xlsx")])>=1)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, datadict: dict):
        """
        Writes cleaned data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the WISE time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dirpath / fname) as writer:
            for name, df in datadict.items():
                df.to_excel(writer, sheet_name=name)

        
class WiseDataImputer(Processor):
    """
    Data imputation functionality.

    Imputing missing values through simple linear
    interpolation will lead to introduction of additional
    spurious correlations (i.e. it would completely defeat
    the point). Therefore, we'll have to resort to a more
    sophisticated data imputation techniques: we'll use
    Gaussian Process (GP) Models to model the time series
    that can be evaluated on a common time grid for all
    time series. GPs are a good choice because they come
    with uncertainty information.

    Class attributes
    ----------------
    c_name : str
        class name = "Data imputer"

    c_description : str
        ultra-short class description = "Impute data (only interpolation, no extrapolation)"

    Private methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """    
    c_name = "Data imputer"
    c_description = "Impute data (only interpolation, no extrapolation)"

    def _transform_single_country(self, df):
        """
        """
        def _determine_imputed(row: pd.Series) -> bool:
            """
            """
            # Step 1: If the value in interp_tracker is null, it can't be interpolated
            if pd.isnull(row["value"]):
                return False
            
            year = row["year"]
            metric = row["metric"]
            
            # Step 2: If value exists in wise_clean, it's not interpolated
            if year in df.index and metric in df.columns:
                wise_value = df.at[year, metric]
                if pd.notnull(wise_value):
                    return False
            
            # Step 3: If wise_clean value is missing but interp_tracker["value"] exists => interpolated
            return True
        
        assert df.index.name == "year"
        
        # Perform data imputation using Gaussian Processes
        di = impute.DataImputer(df)
        di.fit_gp()

        # Read out interpolation results
        wise_interp = di.gp_means
        wise_envlp = di.gp_stds
        wise_interp.index.name = "year"
        wise_envlp.index.name = "year"
        
        # Create book keeping tables that keep track
        # of which values were measured and which
        # ones were imputed.
        interp_tracker = wise_interp.reset_index()\
                                    .rename({"index": "year"}, 
                                            axis=1
                                           )\
                                    .melt(id_vars="year", 
                                          var_name='metric',
                                          value_name='value'
                                         )
        interp_tracker["imputed"] = False
        interp_tracker["method"] = "GPR"
        
        # Apply the logic
        interp_tracker["imputed"] = interp_tracker.apply(_determine_imputed, 
                                                         axis=1
                                                        )

        return wise_interp, wise_envlp, interp_tracker
        
    def _transform(self):
        """
        Impute the data on a common time grid.

        Originally, all time series are measured on
        their very own time grid. In order to make
        them comparable, all these time grids need
        to be aligned. After running through this
        transformation step, every time series will
        have a value for every year between its
        first and last year of measurement.
        """
        wise_interp_dict = dict()
        wise_envlp_dict = dict()
        interp_tracker_dict = dict()
        
        for sheet, df in self.input.items():
            wise_interp, wise_envlp, interp_tracker = self._transform_single_country(df)
            wise_interp_dict[sheet] = wise_interp
            wise_envlp_dict[sheet] = wise_envlp
            interp_tracker_dict[sheet] = interp_tracker

        # Make data available
        self.output = wise_interp_dict
        self.additional_results["uncertainty_envelopes"] = wise_envlp_dict
        self.additional_results["interp_tracker"] = interp_tracker_dict

        # Write processed data to csv files
        self._save(const.interp_data_fname, wise_interp_dict)
        self._save(const.envlp_data_fname, wise_envlp_dict)
        self._save(const.interp_tracker_fname, interp_tracker_dict)

        print("-> done!")
        
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_excel(self.current_stage_fpath / const.interp_data_fname, 
                                    sheet_name=None,
                                    index_col="year"
                                   )
        self.additional_results["uncertainty_envelopes"] = pd.read_excel(self.current_stage_fpath / const.envlp_data_fname, 
                                                                         sheet_name=None,
                                                                         index_col="year"
                                                                        )
            
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if imputed data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.xlsx")])==3)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, datadict: dict):
        """
        Writes imputed data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the WISE time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dirpath / fname) as writer:
            for name, df in datadict.items():
                df.to_excel(writer, sheet_name=name)
        

class WiseTSDecomposer(Processor):
    """
    Take a set of time series (stored as columns
    in a pandas.DataFrame) and decompose it into
    a trend and residuals.

    Class attributes
    ----------------
    c_name : str
        class name = "Time Series Decomposer"

    c_description : str
        ultra-short class description = "Decompose time 
        series into trend and residuals (no seasonality)"

    Private methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """
    c_name = "Time Series Decomposer"
    c_description = "Decompose time series into trend and residuals (no seasonality)"
    
    def _year2date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert integer years to actual dates
        according to the rule

        1950 -> 1950-01-01
        1993 -> 1993-01-01
        2012 -> 2012-01-01

        Parameter
        ---------
        df : pandas.DataFrame
            A pandas.DataFrame containing one or several
            time series in its columns and a row index
            of dtype int.

        Returns
        -------
        df : pandas.DataFrame
            The same pandas.DataFrame as in the input but
            now the row index has an actualy datetime dtype.
        """
        df.index = tsa.fractional_years_to_datetime(df.index)
        return df
        
    def _transform(self):
        """
        Perform time series decomposition.
        """
        residuals_dict = dict()
        trend_dict = dict()
        slt_dict = dict()
        pvalues_dict = dict()

        for sheet, df in self.input.items():
            assert df.index.name == "year"

            ts_data = self._year2date(df.copy())
            ts_analyzer = tsa.TSAnalyzer(ts_data)
            residuals = ts_analyzer.decompose()
            residuals.index.name = "date"

            residuals_dict[sheet] = residuals
            trend_dict[sheet] = ts_analyzer.trend
            slt_dict[sheet] = ts_analyzer.optimal_stl_df.set_index("metric")
            pvalues_dict[sheet] = ts_analyzer.pvalues_df
        
        # Make data available
        self.output = residuals_dict
        self.additional_results["trend"] = trend_dict
        self.additional_results["optimal_stls"] = slt_dict
        self.additional_results["pvalues_df"] = pvalues_dict

        # Write processed data to csv files
        self._save(const.residuals_fname, self.output)
        self._save(const.trends_fname, self.additional_results["trend"])
        self._save(const.optimal_stl_info_fname, self.additional_results["optimal_stls"])
        self._save(const.p_values_fname, self.additional_results["pvalues_df"])

        print("-> done!")
    
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_excel(self.current_stage_fpath / const.residuals_fname, 
                                    parse_dates=["date"], 
                                    sheet_name=None,
                                    index_col="date"
                                   )
        self.additional_results["trends"] = pd.read_excel(self.current_stage_fpath / const.trends_fname, 
                                                          parse_dates=["date"], 
                                                          sheet_name=None,
                                                          index_col="date"
                                                         )
        self.additional_results["p_values"] = pd.read_excel(self.current_stage_fpath / const.p_values_fname, 
                                                            sheet_name=None,
                                                            index_col="metric"
                                                           )
        self.additional_results["optimal_stl"] = pd.read_excel(self.current_stage_fpath / const.optimal_stl_info_fname, 
                                                               sheet_name=None,
                                                               index_col="metric"
                                                              )
        
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if imputed data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.xlsx")])==4)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, datadict: dict):
        """
        Writes resulting data from time series 
        decomposition to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the WISE time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dirpath / fname) as writer:
            for name, df in datadict.items():
                df.to_excel(writer, sheet_name=name)


class WiseDataScaler(Processor):
    """
    Standardize time series data (standardization = subtract
    mean value and scale to unit standard deviation). The
    resulting values are thus z-scores (normally distributed).

    Class attributes
    ----------------
    c_name : str
        class name = "Data scaler"

    c_description : str
        ultra-short class description = "Standard scaling data"

    Private methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """
    c_name = "Data scaler"
    c_description = "Standard scaling data"
    
    def _std_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardized the input data.

        Input data is converted into z-scores.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data

        Returns
        -------
        z_scores : pandas.DataFrame
            Standardized data
        """
        # Initialize and run scaler
        z_scaler = StandardScaler()
        z_scores = z_scaler.fit_transform(data)

        # Turn output into pandas.DataFrame
        z_scores_df = pd.DataFrame(z_scores, index=data.index, columns=data.columns)
        
        return z_scores_df
        
    def _transform(self):
        """
        Perform data scaling.
        """
        normalized_residuals_dict = dict()
        
        for sheet, df in self.input.items():
            assert df.index.name == "date"
        
            normalized_residuals_dict[sheet] = self._std_scale(df)
        

        # Make data available
        self.output = normalized_residuals_dict
        # self.additional_results["normalized_ts"] = normalized_timeseries_dict
        
        # Write processed data to csv files
        self._save(const.scaled_resids_fname, normalized_residuals_dict)
        #self._save(const.scaled_ts_fname, normalized_timeseries_dict)

        print("-> done!")
    
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_excel(self.current_stage_fpath / const.scaled_resids_fname, 
                                    parse_dates=["date"],
                                    sheet_name=None,
                                    index_col="date"
                                   )
                                                       
        #self.additional_results["scaled_time_series"] = pd.read_excel(self.current_stage_fpath / const.scaled_ts_fname,
        #                                                              sheet_name=None,
        #                                                              parse_dates=["date"],
        #                                                              index_col="date"
        #                                                             )

        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if scaled data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.xlsx")])>=1)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, datadict: dict):
        """
        Writes resulting data from time series 
        decomposition to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the WISE time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(dirpath / fname) as writer:
            for name, df in datadict.items():
                df.to_excel(writer, sheet_name=name)

        
class TransformationPipeline(object):
    """
    Put all the data transformation steps into a
    streamlined pipeline, which offers a very clean
    and easy-to-use interface through the "run"
    method.

    The pipeline is built as a stack of all transformation
    stages. The "run" method is built with checkpointing
    and resuming intelligence, i.e. the code checks
    automatically what the latest transformation stage
    stored on disk is, loads that data and only runs
    the remaining steps. A user is, however, able to 
    force the recomputation of transformations via
    the force_stage argument. Notice that all stage
    following the stage passed in the force_stage argument
    will always be recomputed, too. Therefore, setting
    force_stage=1 causes a complete recomputation of all
    data transformation steps.

    Parameters
    ----------
    raw_data : List[Tuple[str, Dict]]
        Contains the raw WISE data as downloaded
        from the www. The tuples are of the format
        (dam_id, excel_spreadsheet), where the excel
        spreadsheet is a dictionary with (sheet_name,
        data table) as key-value pairs.
        
    metatable : pandas.DataFrame
        Contains the metainformation about the each
        metric.

    Optionals
    ---------
    force_stage : int [default = 0]
        Integer (>=0) indicating if at all and if so
        from which stage onwards the data transformation
        is shall be force-recomputed.

        Explanation of behaviour:
        force_stage = 0: no transformation is force-recomputed
        force_stage = i (>0): all stages >=i are force-recomputed

        Notice that this means that if force_stage = 1,
        then all stages are recomputed.
        
    verbosity : int [default = 0] 
        Defines how verbose the output is. 0 means only the bare
        minimum of output is print to std out. The higher the
        value, the more output will be written to std out.

    Attributes
    ----------
    self.raw : List[Tuple[str, Dict]]
        See parameter raw_data.
        
    self.metatable : pandas.DataFrame
        See parameter metatable.
        
    self.stages : List[Processor]
        Contains the stack of all data transformation stages
        (subclasses of Processor class)
        
    self.n_stages : int
        Length of self.stages, number of stages in transformation
        pipeline
        
    self.verbosity : int
        See optional parameter verbosity.

    self.force_stage : int
        See parameter force_stage
        
    self.force_dict : Dict[int, bool]
        A dictionary containing boolean values indicating
        which transformation stages will be force-recomputed.
        Depends on optional parameter force_stage.

        Behavior:
        force_stage = 1 => force_dict = {1: True, 2: True, 3: True, ...}
        force_stage = 2 => force_dict = {1: False, 2: True, 3: True, ...}
        force_stage = 3 => force_dict = {1: False, 2: False, 3: True, ...}

        
    Private methods
    ---------------
    _run_stage(index: int) -> Any

    Public methods
    --------------
    resume() -> pd.DataFrame
        Trigger resuming of transformation pipeline.

    run() -> pd.DataFrame
        Trigger execution of transformation pipeline.
    """
    def __init__(self, 
                 raw_data: pd.DataFrame,
                 metatable: pd.DataFrame,
                 force_stage: int = 0,
                 verbosity: int = 0):
        self.raw_data = raw_data
        self.metatable = metatable

        # Next define the list of data transformation
        # steps:

        # Stage 1: Data reformatting
        #          --> Originally the data for the time series of all
        #              countries are stored in single a long-format 
        #              table. This data reformatting step transforms 
        #              this single table into a dictionary of wide-
        #              format tables where every table contains the
        #              data of a single country only. The columns of
        #              these tables contain the individual WISE metrics
        #              while the rows contain the years.
        # Stage 2: Data cleaning
        #          --> Performs a number of data cleaning steps such
        #              as removal of irrelevant metrics, removal
        #              of time series that are too sparse, etc.
        # Stage 3: Data imputation 
        #          --> converts time series with missing data and
        #              irregular time grids into time series all
        #              having values for every year between their
        #              first and last respective year of measurement.
        # Stage 4: Time series decomposition 
        #          --> converts full time series into residuals, which
        #              can actually be used for data analysis.
        # Stage 5: Scaling
        #          --> normlize time series data
        self.stages = [Stage1(input_data=None, 
                              metatable = self.metatable,
                              stage_id=1, 
                              verbosity=verbosity,
                              ),
                       WiseDataCleaning(input_data=None, 
                                        metatable=self.metatable,
                                        stage_id=2,
                                        verbosity=verbosity
                                       ),
                       WiseDataImputer(input_data=None, 
                                       metatable=self.metatable,
                                       stage_id=3,
                                       verbosity=verbosity
                                      ),
                       WiseTSDecomposer(input_data=None, 
                                        metatable=self.metatable,
                                        stage_id=4,
                                        verbosity=verbosity
                                       ),
                       WiseDataScaler(input_data=None, 
                                      metatable=self.metatable,
                                      stage_id=5,
                                      verbosity=verbosity
                                     )
                      ]
        self.n_stages = len(self.stages)
        self.verbosity = verbosity

        self.force_stage = force_stage
        self.force_dict = {stage_id: False for stage_id in range(1,self.n_stages+1)}
        if force_stage > 0 and force_stage <= self.n_stages:
            for stage_id in range(force_stage, self.n_stages+1):
                self.force_dict[stage_id] = True
        elif force_stage > self.n_stages:
            raise ValueError(f"Value of 'force_stage' is higher than the number of stages in the pipeline ({force_stage}>{self.n_stages}).")

    def _run_stage(self, index: int) -> Any:
        """
        By default, read in the most-processed, available data
        from disk and runs only the remaining stages.
        
        This is a recursive function. By default, the code will
        always check if data corresponding to the output of the
        last processing stage in the "self.stages" stack is available
        on disk. If so, this data will be loaded into memory and the
        function exits. If not, the function keeps checking earlier
        and earlier processing stages in the "self.stages" stack
        recursively, until it finds a stage whose output data is available
        on disk. This is a "checkpoint". The function will then resume
        data transformation from that checkpoint, i.e. it will only
        run the remaining transformation steps.

        Parameters
        ----------
        index : int
            Index of the stage currently being run.

        Returns
        -------
        self.output : Any
            The output of the current transformation stage. The datatype
            of the output critically depends on the transformation stage.
        """
        stage = self.stages[index]

        stage_is_done = stage._is_done()
        force_stage = self.force_dict[index+1]
        
        if stage_is_done and not(force_stage):
            print(f"> Stage {index + 1} ({stage.c_name}) is done. Reading from disk.")
        else:
            if not(stage_is_done):
                print(f"> Stage {index + 1} ({stage.c_name}) not done. ", end="")
            if force_stage:
                print(f"> Forcing recomputation of stage {index + 1}. ", end="")
            if index == 0:
                print("Starting from raw data...")
                stage.input = self.raw_data
            else:
                print("Running prerequisite stage...")
                stage.input = self._run_stage(index - 1)

        if (index == 0 and not(stage_is_done)) or force_stage:
            print(f">> Computing stage {index + 1}...")
        stage.get_data(force=force_stage)
            
        return stage.output

    def resume(self) -> pd.DataFrame:
        """
        Trigger resuming of execution of transformation
        pipeline.

        Returns
        -------
        self.output : pd.DataFrame
            The output of the final transformation stage.
        """
        return self._run_stage(len(self.stages)-1)

    def run(self, stage_id: int|None = None) -> pd.DataFrame:
        """
        Computes or reads the results for each data
        transformation stage.

        Optional
        --------
        stage_id : int [default: None]
            If set, only the stage with id = stage_id
            (1-based) is run (corresponds to a forced
            run).
        
        Returns
        -------
        self.output : pd.DataFrame
            The output of the final transformation stage.
        """
        if stage_id:
            index = stage_id - 1
            self.force_dict[stage_id]=True
            print(f"> Stage {stage_id}:")
            return self._run_stage(index)
        else:
            for index, stage in enumerate(self.stages):
                print(f"> Stage {index + 1}:")
                # Set input
                if index == 0:
                    stage.input = self.raw_data
                else:
                    stage.input = self.stages[index-1].output
    
                # Force re-computations
                force=False
                if (self.force_stage > 0) & (self.force_stage <= index+1):
                    force=True
                stage.get_data(force=force)

            return stage.output

    def collect_results(self) -> Dict[str, pd.DataFrame|List[str]]:
        """
        Collect all the results computed during
        the data transformation into a single
        dictionary.

        Returns
        -------
        all_results : Dict[str, pd.DataFrame|List[str]]
        """
        all_results = {"raw": self.stages[0].output,
                       "clean": self.stages[1].output,
                       "interpolated": self.stages[2].output,
                       "residuals": self.stages[3].output,
                       "normalized_residuals": self.stages[4].output
                      }
        
        for stage in self.stages:
            try:
                for (k,v) in stage.additional_results.items():
                    all_results[k] = v
            except KeyError:
                continue

        print("There results can be accessed via the following keys:")
        for k in all_results.keys():
            print(".", k)
            
        return all_results
        

    def create_inspection_plots(self,
                                create: str|List[str] = 'all',
                                write: bool=True,
                                sort_by_capital: bool=True
                               ) -> List[Tuple[Figure,Axes]]:
        """
        Create plots for visual inspection of data
        transformation pipeline.

        Can create on or several plots depending
        on input instructions.

        Optional Parameters
        -------------------
        create : str|List[str] [default: 'all']
            List of instructions indicating which
            plots should be created. Must be either
            'all' or one value or a combination of
            values from the following list: 
            
            ['clean vs raw', 
             'interpolated vs clean', 
             'trends vs interpolated', 
             'residuals',
             'zscores'
             ]

        write : bool [default: True]
            If True, saves plot(s) to disk

        sort_by_capital: bool [default: True]
            If True, sorts the panels in the resulting
            plot by capital

        Returns
        -------
        figsaxes : List[Tuple[Figure,Axes]]
            The list contains one tuple for every
            entry in the create parameter. Each
            tuple contains a Figure and an Axes
            element.

        Raises
        ------
        ValueError
            If create is of type str but not equal
            to 'all'.

        ValueError
            If create is of type List but any element
            is not in the list

            ['clean vs raw', 
             'interpolated vs clean', 
             'trends vs interpolated', 
             'residuals',
             'zscores']
        """
        wise_metric_metatable = self.metatable.reset_index()
        wise_metric_metatable = wise_metric_metatable[["acronym",
                                                       "metric_acronym", 
                                                       "metric_name",
                                                       "capital - primary"
                                                      ]]\
                                    .drop_duplicates()\
                                    .set_index("acronym")
        
        if sort_by_capital:
            wise_metric_ids_sorted_by_cap = wise_metric_metatable.sort_values(by="capital - primary")
        else:
            wise_metric_ids_sorted_by_cap = wise_metric_metatable

        sorted_indices = [acronym.lower() for acronym in wise_metric_ids_sorted_by_cap.index]
        
        for iso3, df in self.stages[0].output.items():
            wise_data = df.copy()

            wise_data = wise_data[[idx for idx in sorted_indices if idx in wise_data.columns]]
            wise_data.index = pd.to_datetime(wise_data.index.astype(str), 
                                             format="%Y", 
                                             errors="coerce"  # invalid years become NaT
                                            )
            
            max_list = ['clean vs raw', 
                        'interpolated vs clean', 
                        'trends vs interpolated', 
                        'residuals',
                        'zscores']
            
            wise_clean = self.stages[1].output[iso3].copy()
            wise_clean = wise_clean[[idx for idx in sorted_indices if idx in wise_clean.columns]]
            wise_clean.index = pd.to_datetime(wise_clean.index.astype(str), 
                                              format="%Y", 
                                              errors="coerce"  # invalid years become NaT
                                             )
            
            wise_interpolated = self.stages[2].output[iso3].copy()
            wise_interpolated = wise_interpolated[[idx for idx in sorted_indices if idx in wise_interpolated.columns]]
            wise_interpolated.index = tsa.fractional_years_to_datetime(wise_interpolated.index)
            
            wise_envelopes = self.stages[2].additional_results["uncertainty_envelopes"][iso3].copy()
            wise_envelopes = wise_envelopes[[idx for idx in sorted_indices if idx in wise_envelopes.columns]]
            wise_envelopes.index = tsa.fractional_years_to_datetime(wise_envelopes.index)
            
            wise_residuals = self.stages[3].output[iso3].copy()
            wise_residuals = wise_residuals[[idx for idx in sorted_indices if idx in wise_residuals.columns]]
            
            wise_trends = self.stages[3].additional_results["trends"][iso3].copy()
            wise_trends = wise_trends[[idx for idx in sorted_indices if idx in wise_trends.columns]]
            
            wise_zscores = self.stages[4].output[iso3].copy()
            wise_zscores = wise_zscores[[idx for idx in sorted_indices if idx in wise_zscores.columns]]
            
            if isinstance(create, str):
                if create == 'all':
                    create = max_list
                else:
                    raise ValueError("String-typed value of parameter 'create' must be equal to 'all'.")
    
            for action in create:
                if not action in max_list:
                    raise ValueError(f"Action '{action}' is not valid. Must be one or several of {max_list}.")
    
            figsaxes = []
            if 'clean vs raw' in create:
                plotpath = self.stages[1].current_stage_fpath / const.clean_vs_raw_plot_fpath if write else None
                fig, ax = plot.plot_data(wise_clean, 
                                         title = f"{iso3}: Clean vs raw data (clean = line, raw = diamonds)", 
                                         meta_info = wise_metric_metatable,
                                         scatter_df = wise_data,
                                         fpath = plotpath
                                        )
                figsaxes.append((fig, ax))
    
            if 'interpolated vs clean' in create:
                plotpath = self.stages[2].current_stage_fpath / const.interpolated_vs_clean_plot_fpath if write else None
                fig, ax = plot.plot_data(wise_interpolated,
                                         title = f"{iso3}: GP-interpolated vs clean data (interpolated = line, clean = diamonds)",
                                         meta_info = wise_metric_metatable,
                                         scatter_df = wise_clean,
                                         error_df = wise_envelopes,
                                         fpath = plotpath
                                        )
                figsaxes.append((fig, ax))
    
            if 'trends vs interpolated' in create:
                plotpath = self.stages[3].current_stage_fpath / const.trends_vs_interpolated_plot_fpath if write else None
                fig, ax = plot.plot_data(wise_trends,
                                         title = f"{iso3}: Trend lines through GP-interpolated data (diamonds)",
                                         meta_info = wise_metric_metatable,
                                         scatter_df = wise_interpolated,
                                         fpath = plotpath
                                        )
                figsaxes.append((fig, ax))
    
            if 'residuals' in create:
                plotpath = self.stages[3].current_stage_fpath / const.residuals_plot_fpath if write else None
                fig, ax = plot.plot_data(wise_residuals, 
                                         title = f"{iso3}: Residuals after detrending GP-interpolated data",
                                         meta_info = wise_metric_metatable,
                                         fpath = plotpath
                                        )
                figsaxes.append((fig, ax))
    
            if 'zscores' in create:
                plotpath = self.stages[4].current_stage_fpath / const.zscores_plot_fpath if write else None
                fig, ax = plot.plot_data(wise_zscores,
                                         title = f"{iso3}: Normalized residuals of detrended data",
                                         meta_info = wise_metric_metatable,
                                         fpath = plotpath
                                        )
                figsaxes.append((fig, ax))
