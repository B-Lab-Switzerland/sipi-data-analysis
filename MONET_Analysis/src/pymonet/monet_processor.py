 # Std lib imports
import os
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime as dt
from typing import Dict, List, Tuple
from pathlib import Path

# 3rd party imports
import pandas as pd
import numpy as np

# Local imports
from pymonet import monet_aux as aux
from pymonet import monet_consts as const
from pymonet import monet_logger as logger

class Processor(ABC):
    def __init__(self, 
                 previous_stage_data: List[Tuple[str, Dict]], 
                 metatable: pd.DataFrame, 
                 previous_stage_path: str, 
                 current_stage_path: str
                ):
        self.metatable = metatable
        self.damid2obs_map = metatable[["dam_id", "observable"]]\
                                .set_index("dam_id")\
                                .to_dict()["observable"]
        self.previous_stage_fpath = previous_stage_path
        self.current_stage_fpath = current_stage_path
        self.previous_stage_data_list = previous_stage_data
        self.current_stage_data_list = []
        self.log = dict()

    def transform_data(self):
        """Main method to transform data — checks whether to read or compute."""
        if self._is_done():
            self._read()
        else:
            self._transform()
            self._save()
            self._log()
    
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

    def get_data(self, force=False):
        """
        """
        # Read data from disk if it already exists
        
        if (not force) & self._is_done():
            self._read()
    
        # Otherwise, transform the raw data
        else:
            self._transform()
    
class Stage1_Processor(Processor):
    """
    """
    def _return_worksheets(self, raw_spreadsheet):
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

    def _get_table_name(self, table) -> str:
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
        
    def _get_table_description(self, table) -> str:
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

    def _get_dataframe_and_remarks(self, table, description) -> Tuple[pd.DataFrame, str]:
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

    def _trf_single_file(self, dam_id: str, raw_spreadsheet: Dict[str, pd.DataFrame]):
        """
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

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        s1_res : Dict
            Dictionary containing the actual data as a dataframe,
            the description and the remark about the data in that
            dataframe.
        """
        processed_s1_log = logger.MonetLogger(["file_id",
                                               "file_name", 
                                               "file_hash", 
                                               "dam_id", 
                                               "processed_date",
                                               "processed_timestamp"
                                              ])

        for counter, (dam_id, raw_file) in enumerate(self.previous_stage_data_list):
            print(f"Downloading {(counter+1)}/{len(self.previous_stage_data_list)}", end="\r")
            # Process current file
            stage1_data, fname_root, proc_dt = self._trf_single_file(dam_id, raw_file)
            
            # Make data available
            self.current_stage_data_list.append(stage1_data)   

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
        """
        print("Reading stage-1-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.current_stage_fpath / "stage_1").glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.current_stage_data_list.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

    def _is_done(self) -> bool:
        """
        """
        n_files_expected = self.metatable["dam_id"].nunique()
        paths_exist = self.current_stage_fpath.exists()
        print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.json")])==n_files_expected)
        print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname, jsonstr):
        dir_path = self.current_stage_fpath
        aux.json_dump(dir_path, fname, jsonstr)  
             
            
class Stage2_Processor(Processor):
    """
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
        for i, stage1 in enumerate(self.previous_stage_data_list):           
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
    

        self.current_stage_data_list = s2_trafo_results
    
        # Write log files
        processed_s2_log.write(const.log_file_processed_s2_data, index_key="file_id")
        self.log = pd.DataFrame(processed_s2_log.log_dict)

    def _read(self):
        print("Reading stage-2-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.current_stage_fpath).glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.current_stage_data_list.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

    def _is_done(self) -> bool:
        """
        """
        n_dam_ids = self.metatable["dam_id"].nunique()
        paths_exist = self.current_stage_fpath.exists()
        print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.json")])>=n_dam_ids)
        print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname, jsonstr):
        dir_path = self.current_stage_fpath
        aux.json_dump(dir_path, fname, jsonstr)


class Stage3_Processor(Processor):
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
        metrics = [d for d in self.previous_stage_data_list if d["metric_id"].endswith("metr")]
        cis = [d for d in self.previous_stage_data_list if d["metric_id"].endswith("ci")]
        
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
        self.current_stage_data_list = [compact_metrics, compact_cis]

        # Write processed data to csv files
        self._save(const.compact_metrics_filename, compact_metrics)
        self._save(const.compact_cis_filename, compact_cis)
        
        print("-> done!")

    def _read(self):
        print("Reading stage-3-processed data from disk...")
        self.current_stage_data_list.append(pd.read_csv(self.current_stage_fpath / const.compact_metrics_filename))
        self.current_stage_data_list.append(pd.read_csv(self.current_stage_fpath / const.compact_cis_filename))

        self.current_stage_data_list[0].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.current_stage_data_list[0].set_index("Year", inplace=True)
        self.current_stage_data_list[1].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.current_stage_data_list[1].set_index("Year", inplace=True)
        
        print("-> done!")

    def _is_done(self) -> bool:
        """
        """
        paths_exist = self.current_stage_fpath.exists()
        print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in (self.current_stage_fpath).glob("*.csv")])==2)
        print(f"dirs_not_empty: {dirs_not_empty}")

        is_done = paths_exist & dirs_not_empty
        return is_done

    def _save(self, fname, df):
        dirpath = self.current_stage_fpath
        dirpath.mkdir(parents=True, exist_ok=True)
        df.to_csv(dirpath / fname)
        
        