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

from sklearn.preprocessing import StandardScaler

# Local imports
from pymonet import monet_aux as aux
from pymonet import monet_consts as const
from pymonet import monet_logger as logger
from sipi_da_utils import utils

class Processor(ABC):
    """
    Abstract base class from which each data
    transformation step for the MONE 2030 data
    analysis will be subclassed.

    A series of subclasses of the 'Processor' 
    ABC can be used to transform the MONET 2030
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
        metainformation about each MONET
        2030 indicator.

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
                 input_data: Any, 
                 indicator_table: pd.DataFrame|None,
                 metatable: pd.DataFrame, 
                 stage_id: int,
                 verbosity: int
                ):

        # Input data
        self.input = input_data
        self.indicators = indicator_table
        self.metatable = metatable
        self.stage_id = stage_id

        # Further variable declarations
        self.output: Any = None  # This will ultimately store the final result
        self.additional_results: Dict[str, Any] = dict()
        self.previous_stage_fpath: Path = const.processed_dir / f"stage_{stage_id-1}" if stage_id >= 1  else const.raw_dir
        self.current_stage_fpath: Path = const.processed_dir / f"stage_{stage_id}"
        self.log: Dict = dict()
        self.damid2obs_map: pd.DataFrame = metatable[["dam_id", "observable"]]\
                                                .set_index("dam_id")\
                                                .to_dict()["observable"]
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
    Convert raw data to json files.

    The raw data consists of a list of 2-tuples, the first
    element of which is the dam_id of the MONET 2030
    observable and the second is a dictionary representing
    the Excel spreadsheet (.xlsx) scraped from the WWW. These
    spreadsheets are not formatted in an analysis-friendly way.
    This class provides the functionality to convert these
    spreadsheets to json strings that can be used for
    further processing. The resulting json strings have the
    following structure:

    {
     "dam_id": dam_id,
     "observable": observable,
     "description": desc,
     "remark": remark,
     "data": df
    }

    Therefore, the json strings not only contain the actual
    indicator data but metadata as well.

    Private Methods
    ---------------
    _return_worksheets(raw_spreadsheet: Dict[str, pandas.DataFrame]) -> List[str]
    _get_table_name(table: pandas.DataFrame) -> str
    _get_table_description(table: pandas.DataFrame) -> str
    _get_dataframe_and_remarks(table: pandas.DataFrame,
                               description: str
                               ) -> Tuple[pandas.DataFrame, str]
    _trf_single_file(dam_id: str, 
                     raw_spreadsheet: Dict[str, pandas.DataFrame]
                    ) -> Tuple[OrderedDict, str, datetime.datetime]
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    
    Public Methods
    --------------
    get_data(force: bool) -> None
        Is defined in parent ABC "Processor"
    """
    def _return_worksheets(self, raw_spreadsheet: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Returns a list of work sheets in the spreadsheet
        self.raw_spreadsheet.

        Returns
        -------
        worksheets : List[str]
            List of work sheet names.
        """
        worksheets = list(raw_spreadsheet.keys())
        return worksheets

    def _get_table_name(self, table: pd.DataFrame) -> str:
        """
        Extracts the name of the MONET2030
        indicator from the datafile itself.

        The name is located in the top-left
        cell (coordinates 0,0).

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        table_name : str
            Name of the current table/observable
        """
        table_name = table.iloc[0,0]
        return table_name
        
    def _get_table_description(self, table: pd.DataFrame) -> str:
        """
        Extracts the description of the MONET2030
        indicator from the datafile itself.

        The name is located in the cell with
        coordinates (0,1).

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        table_description : str
            Description of the current table/observable
        """
        table_description = table.iloc[1,0]
        return table_description

    def _get_dataframe_and_remarks(self, 
                                   table: pd.DataFrame,
                                   description: str
                                  ) -> Tuple[pd.DataFrame, str]:
        """
        Extracts the actual data (as a dataframe)
        and possibly a remark from the table.

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        description : str
            Description of the current table/observable

        Returns
        -------
        df : pandas.DataFrame
            DataFraem containing the actual data for the
            current observable.

        remark : str
            Remarks found at the end of the spreadsheet.
        """
        # Check if there actually is a description
        # or whether it's missing
        if description is np.nan:
            column_headers_row = 2
        else:
            column_headers_row = 3

        # Get all the column names for the current data set
        # They follow after the description and a blank line
        col_names = [v for v in table.iloc[column_headers_row,:].values if (v is not np.nan and len(v.strip())>0)]

        # Extract the data (at this point including additional
        # information and footers)
        df = table.iloc[column_headers_row+1:,:]

        # Now let's put in the column names we extracted
        # above.
        col_rename_dict = dict()
        col_rename_dict[df.columns[0]] = "Year"
        for i in range(1, len(col_names)+1):
            col_rename_dict[df.columns[i]] = col_names[i-1]
        df = df.rename(col_rename_dict, axis=1)
        df = df.set_index("Year")

        # Figure out where the actual data
        # ends and where the footer starts
        for cntr, idx in enumerate(df.index):
            if idx!=idx:
                stop = cntr
                break

        # Use this info to extract the remarks (footer)
        remark = " ".join([str(txt) for txt in df.index[stop:] if txt==txt])

        # Separate the actual data
        df = df.iloc[:stop,:]

        return df, remark

    def _trf_single_file(self, 
                         dam_id: str, 
                         raw_spreadsheet: Dict[str, pd.DataFrame]
                        ) -> Tuple[OrderedDict, str, dt]:
        """
        Extracts meta information from spreadsheet and
        singles out data.

        Each spreadsheet does not only contain the numeric indicator
        data but also additional meta information such as the name
        of the indicator, a description, potentially a remark etc.
        All this meta information must be stored separately from the
        data for it to be analysis-friendly. This separation is
        achieved in this method.

        Parameters
        ----------
        dam_id : str
            dam_id of the MONET 2030 observable reported on in
            the variable raw_spreadsheet.
            
        raw_spreadsheet : Dict[str, pd.DataFrame]
            Excel spreadsheet in the form of a python dictionary
            (i.e. key-value pairs). The keys correspond to the
            name of each worksheet and the values contain the
            actual table content.
            
        Returns
        -------
        stage1_data : OrderedDict
            Dictionary containing the data and meta data of
            the current indicator.
        
        file_name_root : str
            File name root used to store the processed version
            of the current data.
            
        proc_dt : dt
            Python timestamp at which the stage 1 data processing
            of the current spreadsheet was completed.
        """
        observable = self.damid2obs_map[int(dam_id)]
        file_name_root = f"m2030ind_damid_{dam_id.zfill(8)}"
        
        # Get a list of work sheets in the spreadsheet
        # self.raw_spreadsheet
        sheetnames = self._return_worksheets(raw_spreadsheet)

        # Extract the actual data table (which is the first sheet)
        table = raw_spreadsheet[sheetnames[0]]

        # Get the name
        name = self._get_table_name(table)

        # Get the description
        desc = self._get_table_description(table)

        # Get the actual data table and remarks
        df, remark = self._get_dataframe_and_remarks(table, desc)

        # Give the columns a name
        df.columns.name = name

        # Collect the stage-1-transformation results and
        # return
        stage1_data = OrderedDict({"dam_id": dam_id,
                                   "observable": observable,
                                   "description": desc,
                                   "remark": remark,
                                   "data": df
                                  })
        proc_dt = dt.now(tz=const.zurich_tz)

        return stage1_data, file_name_root, proc_dt

    def _transform(self):
        """
        Performs the stage 1 transformation of the
        MONET2030 data.

        Raises
        ------
        TypeError
            If self.input is not of type 'list'.
        """
        # Expects: input_data is List[Tuple[str, Dict[str, pd.DataFrame]]], i.e.
        # a list of 2-tuples each of which contains a dam_id as the first entry
        # and an actual dataframe/table as the second entry.
        if not isinstance(self.input, list):
            raise TypeError(f"self.input is of type '{type(self.input)}' but should be of type 'list'.")
        self.output = []

        # Setup logger
        processed_s1_log = logger.MonetLogger(["file_id",
                                               "file_name", 
                                               "file_hash", 
                                               "dam_id", 
                                               "processed_date",
                                               "processed_timestamp"
                                              ])

        for counter, (dam_id, raw_file) in enumerate(self.input):
            # Process current file
            stage1_data, fname_root, proc_dt = self._trf_single_file(dam_id, raw_file)
            
            # Make data available
            self.output.append(stage1_data)   

            # Serialize the processed data to json string
            serializable_data = {k: aux.serialize_value(v) for k, v in stage1_data.items()}

            # Write data to disk
            self._save(fname_root + ".json", serializable_data)
        
            # Collect logging info
            processed_hash = aux.json_hasher(json.dumps(serializable_data))
            processed_s1_log.update({"file_id": f"{dam_id}_p",
                                     "file_name": fname_root+".json", 
                                     "file_hash": processed_hash, 
                                     "dam_id": dam_id, 
                                     "processed_date": proc_dt.strftime(format="%Y-%m-%d"),
                                     "processed_timestamp": proc_dt.strftime(format="%H:%M:%S")})
    
            # Write log files
            processed_s1_log.write(const.log_file_processed_s1_data, index_key="file_id")
            self.log = pd.DataFrame(processed_s1_log.log_dict)
            
    def _read(self):
        """
        Read stage 1 data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading stage-1-processed data from disk...")

        self.output = []
        sorted_json_files = sorted([file for file in (self.current_stage_fpath).glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.output.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
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
        n_files_expected = self.metatable["dam_id"].nunique()
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.json")])==n_files_expected)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, jsonstr: str):
        """
        Writes stage-1-processed data (i.e. a json string)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        jsonstr : str
            JSON string containing the processed data.
        """
        dir_path = self.current_stage_fpath
        aux.json_dump(dir_path, fname, jsonstr)  
             
            
class Stage2(Processor):
    """
    Convert stage-1 JSON files to stage-2 JSON files.

    The JSON files created in stage 1 still contain data fields
    that potentially have multiple columns. If this is the case, 
    this means that the MONET 2030 observable has actually several
    sub-observables (called "metrics"). Stage 2 of the data
    processing pipeline creates a separate JSON file for each
    metric.
    
    The resulting json strings have the
    following structure:

    {
     "metric_id": str,
     "dam_id": str,
     "observable": str,
     "description": str, 
     "remark": str,
     "type": "metric"|"confidence interval",
     "data": pandas.DataFrame
    }

    Therefore, the stage-2 JSON strings not only contain the actual
    metric data but metadata as well.

    Private Methods
    ---------------
    _create_metric_dfs(obs_df: pandas.DataFrame) -> List[pandas.DataFrame]
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    
    Public Methods
    --------------
    get_data(force: bool) -> None
        Is defined in parent ABC "Processor"
    """
    def _create_metric_dfs(self, obs_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        From a MONET2030 observable dataframe that 
        contains several columns (subobservables), 
        create one dataframe per metric/
        column.

        Parameters
        ----------
        obs_df : pandas.DataFrame
            DataFrame containing all the data
            for a specific MONET2030 observable.

        Returns
        -------
        subobs_df_list : List[pandas.DataFrame]
            List of single-column dataframes.
            Each dataframe contains data about
            one specific metric.
        """

        subobs_df_list = [] # metric
        ci95_df_list = []   # 95% confidence intervals
    
        # Iterate over all columns and create
        # a separate dataframe for each one
        for col_idx, col in enumerate(obs_df.columns):
            if "confidence interval" in col.lower():
                # Rename the column to integrate the actual observable name
                new_col_name = f"{col} ({obs_df.columns[col_idx-1]})"
                new_df = pd.DataFrame(index = obs_df.index, columns = [new_col_name])
                new_df[new_col_name] = obs_df.iloc[:, col_idx]
                ci95_df_list.append(new_df)
            else:
                subobs_df_list.append(obs_df[[col]])

        return subobs_df_list, ci95_df_list

    def _create_id2name_map(self, json_dict) -> pd.DataFrame:
        """
        Create a map from metric_ids to metric names.

        Parameters
        ----------
        json_dict : Dict
            Dictionary corresponding to the stage-2 JSON
            string for each MONET2030 metric.

        Returns
        -------
        id2name_map : pandas.DataFrame
            Dataframe mapping the metric IDs to the metric
            names (including some descriptions).
        """
        id2name_map = pd.DataFrame([{"metric_id": od["metric_id"], 
                                     "metric_name": f"{od["observable"]} [{od["data"].columns[0]}]", 
                                     "metric_description": od["description"]
                                    } for od in json_dict
                                   ]
                                  )

        return id2name_map.set_index("metric_id")

    def _transform(self):
        """
        Performs the stage 2 transformation of the
        MONET2030 data.
        """
        processed_s2_log = logger.MonetLogger(["file_id",
                                               "file_name", 
                                               "file_hash", 
                                               "dam_id", 
                                               "metric_id",
                                               "processed_date",
                                               "processed_timestamp"
                                              ])
    
        # Iterate over all columns and create
        # a separate dataframe for each one

        s2_trafo_results = []
        for i, stage1 in enumerate(self.input):           
            metric_df_list, ci95_df_list = self._create_metric_dfs(stage1["data"])
            
            for subobs_id, subobs in enumerate(metric_df_list):
                metric_dict = OrderedDict({"metric_id": str(stage1["dam_id"]).zfill(8) + chr(97+subobs_id)+"_metr",
                                           "dam_id": stage1["dam_id"],
                                           "observable": stage1["observable"],
                                           "description": stage1["description"], 
                                           "remark": stage1["remark"],
                                           "type": "metric",
                                           "data": subobs
                                          })
                proc_dt = dt.now(tz=const.zurich_tz)

                s2_trafo_results.append(metric_dict)

                serializable_metric = {k: aux.serialize_value(v) for k, v in metric_dict.items()}

                # Write processed data to json file
                fname = f"m2030ind_damid_{metric_dict["metric_id"]}.json"
                self._save(fname, serializable_metric)

                processed_hash = aux.json_hasher(json.dumps(serializable_metric))
                processed_s2_log.update({"file_id": f"{metric_dict["metric_id"]}",
                                         "file_name": fname, 
                                         "file_hash": processed_hash, 
                                         "dam_id": stage1["dam_id"], 
                                         "metric_id": metric_dict["metric_id"],
                                         "processed_date": proc_dt.strftime(format="%Y-%m-%d"),
                                         "processed_timestamp": proc_dt.strftime(format="%H:%M:%S")})
                
            for ci_id, ci in enumerate(ci95_df_list):
                ci_dict = OrderedDict({"metric_id": str(stage1["dam_id"]).zfill(8) + chr(97+ci_id)+"_ci",
                                       "dam_id": stage1["dam_id"],
                                       "observable": stage1["observable"],
                                       "description": stage1["description"], 
                                       "remark": stage1["remark"],
                                       "type": "confidence interval",
                                       "data": ci
                                    })
                proc_dt = dt.now(tz=const.zurich_tz)
                
                s2_trafo_results.append(ci_dict)

                serializable_ci = {k: aux.serialize_value(v) for k, v in ci_dict.items()}

                # Write processed data to json file
                fname = f"m2030ind_damid_{ci_dict["metric_id"]}.json"
                self._save(fname, serializable_ci)

                # Collect logging info
                processed_hash = aux.json_hasher(json.dumps(serializable_ci))
                processed_s2_log.update({"file_id": f"{ci_dict["metric_id"]}",
                                         "file_name": fname, 
                                         "file_hash": processed_hash, 
                                         "dam_id": stage1["dam_id"], 
                                         "metric_id": ci_dict["metric_id"],
                                         "processed_date": proc_dt.strftime(format="%Y-%m-%d"),
                                         "processed_timestamp": proc_dt.strftime(format="%H:%M:%S")})
    

        metric_id2name_map = self._create_id2name_map(s2_trafo_results)
        # Save id-to-name map
        metric_id2name_map.to_csv(self.current_stage_fpath / const.metric_id2name_fname)
        
        # Make data available
        self.output = s2_trafo_results
        self.additional_results["metric_id2name_map"] = metric_id2name_map
    
        # Write log files
        processed_s2_log.write(const.log_file_processed_s2_data, index_key="file_id")
        self.log = pd.DataFrame(processed_s2_log.log_dict)

    def _read(self):
        """
        Read stage 2 data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading stage-2-processed data from disk...")
        self.output = []
        sorted_json_files = sorted([file for file in (self.current_stage_fpath).glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.output.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if stage 2 data is already available on disk
        or not.

        Returns
        -------
        is_done : bool
            True if data is available on disk. False if it
            is not.
        """
        n_dam_ids = self.metatable["dam_id"].nunique()
        paths_exist = self.current_stage_fpath.exists()

        if self.verbosity > 0:
            print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.json")])>=n_dam_ids)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname, jsonstr):
        """
        Writes stage-2-processed data (i.e. a json string)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        jsonstr : str
            JSON string containing the processed data.
        """
        dir_path = self.current_stage_fpath
        aux.json_dump(dir_path, fname, jsonstr)


class Stage3(Processor):
    """
    Consolidates the time series and confidence
    interval data stored in several JSON files
    (after stage 2) into two dataframes (one for
    the time series data, one for the confidence
    intervals).

    Each time series and confidence intervals set
    translates to a column in the respective data-
    frame. The rows of the dataframes correspond
    to the year in which the metric/confidence
    interval was measured.

    These dataframes do not contain any meta-data.

    Private Methods
    ---------------
    _standardize_colnames(df: pandas.DataFrame, metric_id: str|List[str]) -> pandas.DataFrame
    _integerize_year_ranges(df: pandas.DataFrame) -> pandas.DataFrame:
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool

    Public Methods
    --------------
    compactify() -> Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    def _standardize_colnames(self, df: pd.DataFrame, metric_id: str|List[str]) -> pd.DataFrame:
        """
        Standardizes column names by mapping the current column
        name to the corresponding unique metric ID.
        """
        df = df.copy()
        
        # harmonize data types
        if isinstance(metric_id,str):
            metric_id = [metric_id]

        if len([c for c in df.columns])!=len(metric_id):
            display(df)
            print("df.columns:", df.columns)
            print("metric_id", metric_id)
            raise ValueError("len(df) and len(metric_id) must be identical, or, "
                             + "if metric_id is a string, then df can only have a "
                             + "single column."
                            )

        for col, mid in zip(df.columns, metric_id):
            df.rename({col: mid}, axis=1, inplace=True)

        return df.copy()
        
    def _integerize_year_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integerizes year ranges, i.e. year ranges such as
        "1996/2000" are mapped to 1996.
        """
        df = df.copy()
        if any(["/" in str(idx) for idx in df.index]):
            df.index = [int(idx.split("/")[0].strip()) for idx in df.index]
        
        df.index = df.index.astype(int)

        return df
        
    def compactify(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compactify all metrics from stage-2-processing
        into a single table.
        """
        # Split metrics from confidence intervals
        metrics = [d for d in self.input if d["metric_id"].endswith("metr")]
        cis = [d for d in self.input if d["metric_id"].endswith("ci")]
        
        metr_df_list = []
        ci_df_list = []
        for metric_dict in metrics:
            data = metric_dict["data"]
            data = self._standardize_colnames(data, metric_dict["metric_id"])
            data = self._integerize_year_ranges(data)
            metr_df_list.append(data.copy())

        for ci_dict in cis:
            data = ci_dict["data"]
            data = self._standardize_colnames(data, ci_dict["metric_id"])
            data = self._integerize_year_ranges(data)
            ci_df_list.append(data)
        
        compact_metric_df = metr_df_list[0]
        for df in metr_df_list[1:]:
            if df.columns[0] in compact_metric_df.columns:
                continue
            compact_metric_df = compact_metric_df.merge(df, how="outer", left_index=True, right_index=True)

        compact_ci_df = ci_df_list[0]
        for df in ci_df_list[1:]:
            if df.columns[0] in compact_ci_df.columns:
                continue
            compact_ci_df = compact_ci_df.merge(df, how="outer", left_index=True, right_index=True)

        return compact_metric_df, compact_ci_df

    def _transform(self):
        """
        Performs the stage 3 transformation of the
        MONET2030 data.

        The actual data from the stage-2 JSON files
        is consolidated into two pandas.DataFrames:
        one containing the actual metric values, the
        other containing 95% confidence intervals.
        In this step all meta-information is ignored.
        In both dataframes each column refers to a
        specific metric and each row to a year.
        """
        compact_metrics, compact_cis = self.compactify()
        
        # Make data available
        self.output = compact_metrics
        self.additional_results["confidence_intervals"] = compact_cis

        # Write processed data to csv files
        self._save(const.compact_metrics_filename, compact_metrics)
        self._save(const.compact_cis_filename, compact_cis)
        
        print("-> done!")

    def _read(self):
        """
        Read stage 3 data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading stage-3-processed data from disk...")
            
        metr_df = pd.read_csv(self.current_stage_fpath / const.compact_metrics_filename)
        ci_df = pd.read_csv(self.current_stage_fpath / const.compact_cis_filename)

        metr_df.rename({"Unnamed: 0": "year"}, axis=1, inplace=True)
        metr_df.set_index("year", inplace=True)
        ci_df.rename({"Unnamed: 0": "year"}, axis=1, inplace=True)
        ci_df.set_index("year", inplace=True)
        
        self.output = metr_df
        self.additional_results["confidence_intervals"] = ci_df
        
        print("-> done!")

    def _is_done(self) -> bool:
        """
        Checks if stage 3 data is already available on disk
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
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])==2)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname, df):
        """
        Writes stage-3-processed data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the MONET2030 time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)


class DataCleaning(Processor):
    """
    Data cleaning functionality.

    Run through a set of different data cleaning
    steps as explained in more detail in the
    docstring of the _transform method.

    Private methods
    ---------------
    _find_relevant_metrics(metrics_df: pandas.DataFrame) -> pandas.DataFrame
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """
    def _find_relevant_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for metrics that are relevant for agenda2030.

        The list of MONET2030 indicators contains a flag whether
        or not the indicators are agenda2030-relevant. Based on
        this list, this method infers if the corresponding observables
        and ultimately metrics are agend2030-relevant or not.

        Parameters
        ----------
        metrics_df : pandas.DataFrame
            A pandas.DataFrame containing all the MONET2030 metrics
            (time series) as columns and the years at which these
            time series were measured as rows.

        Returns
        -------
        relevant_metrics_df : pandas.DataFrame
            A pandas.DataFrame containing the time series of only
            the agenda2030-relevant metrics.

        Raises
        ------
        ValueError
            If the consistency check fails (implying a bug).
        """
        # Merge the indicator table with the meta table (observables-level)
        # in order to map which observables are agenda2030-relevant
        complete_meta_df = self.indicators.merge(self.metatable, left_on="id", right_on="indicator_id")

        # Now filter out all observables that are *not* agenda2030-relevant
        irrelevant_observables = set(complete_meta_df.loc[complete_meta_df["agenda2030_relevant"]==0, "dam_id"].values)

        # Based on the irrelevant observables, derive the irrelevant metrics
        irrelevant_metrics = [c for c in metrics_df.columns if int(c.split("_")[0][:-1]) in irrelevant_observables]

        # Consistency check
        if set([int(m.split("_")[0][:-1]) for m in irrelevant_metrics]) != set(irrelevant_observables):
            raise ValueError("Mismatch in irrelevant observables --> There is a bug in the code of the _find_irrelevant_metrics method!")

        # Drop all those irrelevant metrics, resulting in a dataframe
        # that only contains the relevant ones.¨
        relevant_metrics_df = metrics_df.drop(irrelevant_metrics, axis=1).copy()
        return relevant_metrics_df
        
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
        # Perform dataset-specific cleaning
        relevant_metrics_df = self._find_relevant_metrics(self.input)

        # Perform generic cleaning steps
        cleaner = utils.DataCleaner(relevant_metrics_df, verbose=self.verbosity)
        duplicated_rows = cleaner.drop_duplicates()
        constant_cols = cleaner.remove_constant_columns()
        outside_years = cleaner.apply_time_filter(max_year = 2025)
        sparse_cols = cleaner.drop_sparse_columns(n_notnull_min = 10)

        # Make data available
        cleaner.df.index.name = "year"
        self.output = cleaner.df

        # Write processed data to csv files
        self._save(const.clean_data_fname, cleaner.df)
        
    def _read(self):
        """
        Read clean data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading clean data from disk...")

        self.output = pd.read_csv(self.current_stage_fpath / const.clean_data_fname).set_index("year")
            
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
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])==1)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, df: pd.DataFrame):
        """
        Writes cleaned data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the MONET2030 time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname) 


class DataImputer(Processor):
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

    Private methods
    ---------------
    _transform() -> None
    _read() -> None
    _save() -> None
    _is_done() -> bool
    """    
    def _determine_imputed(self, row: pd.Series) -> bool:
        # Step 1: If the value in interp_tracker is null, it can't be interpolated
        if pd.isnull(row["value"]):
            return False
        
        year = row["year"]
        metric = row["metric"]
        
        # Step 2: If value exists in monet_clean, it's not interpolated
        if year in self.input.index and metric in self.input.columns:
            monet_value = self.input.at[year, metric]
            if pd.notnull(monet_value):
                return False
        
        # Step 3: If monet_clean value is missing but interp_tracker["value"] exists => interpolated
        return True
        
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
        # Perform data imputation using Gaussian Processes
        di = utils.DataImputer(self.input)
        di.fit_gp()

        # Read out interpolation results
        monet_interp = di.gp_means
        monet_envlp = di.gp_stds
        
        # Create book keeping tables that keep track
        # of which values were measured and which
        # ones were imputed.
        interp_tracker = monet_interp.reset_index()\
                                   .rename({"index": "year"}, axis=1)\
                                   .melt(id_vars="year", 
                                         var_name='metric',
                                         value_name='value'
                                        )
        interp_tracker["imputed"] = False
        interp_tracker["method"] = "GPR"
        
        # Apply the logic
        interp_tracker["imputed"] = interp_tracker.apply(self._determine_imputed, axis=1)

        # Make data available
        self.output = monet_interp
        self.output.index.name = "year"
        self.additional_results["uncertainty_envelopes"] = monet_envlp
        self.additional_results["uncertainty_envelopes"].index.name = "year"
        self.additional_results["interp_tracker"] = interp_tracker

        # Write processed data to csv files
        self._save(const.interp_data_fname, monet_interp)
        self._save(const.envlp_data_fname, monet_envlp)
        self._save(const.interp_tracker_fname, interp_tracker)
        
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_csv(self.current_stage_fpath / const.interp_data_fname).set_index("year")
        self.additional_results["uncertainty_envelopes"] = pd.read_csv(self.current_stage_fpath / const.envlp_data_fname).set_index("year")
            
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
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])==3)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, df: pd.DataFrame):
        """
        Writes imputed data (i.e. a pandas.DataFrame)
        to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the MONET2030 time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)

        
class TSDecomposer(Processor):
    """
    Take a set of time series (stored as columns
    in a pandas.DataFrame) and decompose it into
    a trend and residuals.
    """
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
        df.index = utils.fractional_years_to_datetime(df.index)
        return df
        
    def _transform(self):
        """
        Perform time series decomposition.
        """
        ts_data = self._year2date(self.input.copy())
        tsa = utils.TSAnalyzer(ts_data)
        residuals = tsa.decompose()

        # Make data available
        self.output = tsa.residuals
        self.additional_results["trend"] = tsa.trend
        self.additional_results["optimal_stls"] = tsa.optimal_stl_df.set_index("metric")
        self.additional_results["pvalues_df"] = tsa.pvalues_df

        # Write processed data to csv files
        self._save(const.residuals_fname, tsa.residuals)
        self._save(const.trends_fname, tsa.trend)
        self._save(const.optimal_stl_info_fname, tsa.optimal_stl_df)
        self._save(const.p_values_fname, tsa.pvalues_df)

        print("-> done!")
    
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_csv(self.current_stage_fpath / const.residuals_fname).set_index("date")
        self.additional_results["trends"] = pd.read_csv(self.current_stage_fpath / const.trends_fname).set_index("date")
        self.additional_results["p_values"] = pd.read_csv(self.current_stage_fpath / const.p_values_fname).set_index("metric")
        self.additional_results["optimal_stl"] = pd.read_csv(self.current_stage_fpath / const.optimal_stl_info_fname).set_index("metric")
        
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
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])==4)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, df: pd.DataFrame):
        """
        Writes resulting data from time series 
        decomposition to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the MONET2030 time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)

class DataScaler(Processor):
    """
    Standardize time series data (standardization = subtract
    mean value and scale to unit standard deviation). The
    resulting values are thus z-scores (normally distributed).
    """
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
        normalized_residuals = self._std_scale(self.input)
        

        # Make data available
        self.output = normalized_residuals
        # self.additional_results["normalized_ts"] = normalized_timeseries
        
        # Write processed data to csv files
        self._save(const.scaled_resids_fname, normalized_residuals)
        #self._save(const.scaled_ts_fname, normalized_timeseries)

        print("-> done!")
    
    def _read(self):
        """
        Read imputed data from disk if available.
        """
        if self.verbosity > 0:
            print("Reading imputed data from disk...")

        self.output = pd.read_csv(self.current_stage_fpath / const.scaled_resids_fname).set_index("date")
        #self.additional_results["scaled_time_series"] = pd.read_csv(self.current_stage_fpath / const.scaled_ts_fname).set_index("date")

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
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])>=1)

        if self.verbosity > 0:
            print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname: str, df: pd.DataFrame):
        """
        Writes resulting data from time series 
        decomposition to disk.

        Parameters
        ----------
        fname : str
            Name of the file the data is written to.

        df : pandas.DataFrame
            DataFrame containing the MONET2030 time series data.
        """
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)
    
    
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
        Contains the raw MONET2030 data as downloaded
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
                 raw_data: List[Tuple[str, Dict]],
                 indicator_table: pd.DataFrame,
                 metatable: pd.DataFrame,
                 force_stage: int = 0,
                 verbosity: int = 0):
        self.raw_data = raw_data
        self.metatable = metatable

        # Next define the list of data transformation
        # steps:

        # Stage 1: Separation of data and meta information
        #          --> Converts xlsx files containing both 
        #              time series data an meta information
        #              into JSON files that cleanly separate
        #              the meta information from the data.
        # Stage 2: Creation of individual metrics
        #          --> Creates JSON files that one and only one
        #              time series (i.e. in some cases a JSON
        #              file resulting from stage 1 is split into
        #              multiple JSON files here in stage 2 if the
        #              stage-1 file had a data table with multiple
        #              columns).
        # Stage 3: Data consolidation
        #          --> Takes all the time series in the JSON files
        #              from stage-2 and consolidate them into 
        #              pandas.DataFrames. In this step any meta
        #              information is neglected. All metrics end
        #              up in one DataFrame, all confidence intervals
        #              in another.
        # Stage 4: Data cleaning
        #          --> Performs a number of data cleaning steps such
        #              as removal of irrelevant metrics, removal
        #              of time series that are too sparse, etc.
        # Stage 5: Data imputation 
        #          --> converts time series with missing data and
        #              irregular time grids into time series all
        #              having values for every year between their
        #              first and last respective year of measurement.
        # Stage 6: Time series decomposition 
        #          --> converts full time series into residuals, which
        #              can actually be used for data analysis.
        # Stage 7: Scaling
        #          --> normlize time series data
        self.stages = [Stage1(input_data=None, 
                              indicator_table = indicator_table,
                              metatable=self.metatable, 
                              stage_id=1, 
                              verbosity=verbosity
                             ),
                       Stage2(input_data=None, 
                              indicator_table = indicator_table,
                              metatable=self.metatable, 
                              stage_id=2, 
                              verbosity=verbosity
                             ),
                       Stage3(input_data=None, 
                              indicator_table = indicator_table,
                              metatable=self.metatable,
                              stage_id=3,
                              verbosity=verbosity
                             ),
                       DataCleaning(input_data=None, 
                                    indicator_table = indicator_table,
                                    metatable=self.metatable,
                                    stage_id=4,
                                    verbosity=verbosity
                                   ),
                       DataImputer(input_data=None, 
                                   indicator_table = indicator_table,
                                   metatable=self.metatable,
                                   stage_id=5,
                                   verbosity=verbosity
                                  ),
                       TSDecomposer(input_data=None,
                                    indicator_table = indicator_table,
                                    metatable=self.metatable,
                                    stage_id=6,
                                    verbosity=verbosity
                                   ),
                       DataScaler(input_data=None,
                                  indicator_table = indicator_table,
                                  metatable=self.metatable,
                                  stage_id=7,
                                  verbosity=verbosity
                                 ),
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
            print(f"> Stage {index + 1} is done. Reading from disk.")
        else:
            if not(stage_is_done):
                print(f"> Stage {index + 1} not done. ", end="")
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

    def run(self) -> pd.DataFrame:
        """
        Computes or reads the results for each data
        transformation stage.

        Returns
        -------
        self.output : pd.DataFrame
            The output of the final transformation stage.
        """
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