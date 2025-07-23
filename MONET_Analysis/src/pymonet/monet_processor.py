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

# Local imports
from pymonet import monet_aux as aux
from pymonet import monet_consts as const
from pymonet import monet_logger as logger

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
                 metatable: pd.DataFrame, 
                 stage_id: int,
                 verbosity: int
                ):

        # Input data
        self.input = input_data
        self.metatable = metatable
        self.stage_id = stage_id

        # Further variable declarations
        self.output: Any = None  # This will ultimately store the final result
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
        print("Reading stage-1-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.current_stage_fpath / "stage_1").glob("*.json")])
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

    def _transform(self):
        """
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
    

        self.output = s2_trafo_results
    
        # Write log files
        processed_s2_log.write(const.log_file_processed_s2_data, index_key="file_id")
        self.log = pd.DataFrame(processed_s2_log.log_dict)

    def _read(self):
        print("Reading stage-2-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.current_stage_fpath).glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.output.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

    def _is_done(self) -> bool:
        """
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
        dir_path = self.current_stage_fpath
        aux.json_dump(dir_path, fname, jsonstr)


class Stage3(Processor):
    """
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
        """
        compact_metrics, compact_cis = self.compactify()
        
        # Make data available
        self.output = [compact_metrics, compact_cis]

        # Write processed data to csv files
        self._save(const.compact_metrics_filename, compact_metrics)
        self._save(const.compact_cis_filename, compact_cis)
        
        print("-> done!")

    def _read(self):
        print("Reading stage-3-processed data from disk...")
        self.output = [pd.read_csv(self.current_stage_fpath / const.compact_metrics_filename),
                       pd.read_csv(self.current_stage_fpath / const.compact_cis_filename)
                      ]

        self.output[0].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.output[0].set_index("Year", inplace=True)
        self.output[1].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.output[1].set_index("Year", inplace=True)
        
        print("-> done!")

    def _is_done(self) -> bool:
        """
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
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)
        

class TransformationPipeline:
    """
    """
    def __init__(self, 
                 raw_data: List[Tuple[str, Dict]],
                 metatable: pd.DataFrame,
                 force: int|bool = False,
                 verbosity: int = 0):
        self.raw_data = raw_data
        self.metatable = metatable
        self.force = force
        self.stages = [Stage1(input_data=None, metatable=self.metatable, stage_id=1, verbosity=verbosity),
                       Stage2(input_data=None, metatable=self.metatable, stage_id=2, verbosity=verbosity),
                       Stage3(input_data=None, metatable=self.metatable, stage_id=3, verbosity=verbosity),
                      ]
        self.n_stages = len(self.stages)

        if not(self.force in set(list(range(1,self.n_stages+1))+[True, False])):
            raise ValueError(f"Parameter 'force' must either be a boolean or an integer between 1 and {self.n_stages+1}.")

    def _run_stage(self, index: int):
        """
        """
        stage = self.stages[index]

        stage_is_done = stage._is_done()
        force_stage = self.force if isinstance(self.force, bool) else self.force==index+1
        
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

        stage.get_data(force=force_stage)
            
        return stage.output

    def run(self):
        """
        """
        return self._run_stage(len(self.stages)-1)