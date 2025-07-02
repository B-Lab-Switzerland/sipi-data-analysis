# Std lib imports
import errno
import os
import json
from collections import OrderedDict
from datetime import datetime as dt
from typing import Dict
from pathlib import Path

# 3rd party imports
import pandas as pd
import numpy as np

# Local imports
from pymonet import monet_etl as etl
from pymonet import monet_aux as aux
from pymonet import monet_consts as const

class IndicatorTableLoader(object):
    """
    Class handling loading MONET2030 indicator data
    into memory.

    Parameters
    ----------
    indicator_table_url : str
        URL pointing to the indicator table

    indicator_table_path : str
        Path where the indicator table is stored
        or will be written to.
    """
    def __init__(self, indicator_table_url: str, indicator_table_path: str):
        self.url = indicator_table_url
        self.fpath = indicator_table_path
        self.table = None
        
    async def _scrape_table(self):
        """
        Scrapes the indicator table from the WWW.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        print("Scraping...")
        # ETL process for Monet2030 indicator list
        etl_mil = etl.ETL_MonetIndicatorList(self.url)
        await etl_mil.extract()
        etl_mil.transform()
        print("-> done!")

        self.table = etl_mil.df

        # Write table to file
        Path("/".join(self.fpath.as_posix().split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        self.table.to_csv(self.fpath)

        # Unset index (if any)
        self.table = self.table.reset_index()

        
    def _read_table(self) -> pd.DataFrame:
        """
        Reads MONET2030 indicator tabel from disk, assuming the
        file exists at the location indicated by self.fpath.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        Raises
        ------
        FileNotFoundError
            If the file cannot be found at self.fpath
        """
        if not self.fpath.exists():
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT),
                                    self.fpath)
            
        print("Reading from disk...")
        self.table = pd.read_csv(self.fpath)
        if "index" in self.table.columns:
            self.table.drop("index", axis=1, inplace=True)
        print("-> done!")
    
    async def get_table(self, force_download=False):
        """
        Reads MONET2030 indicator table into memory.
    
        Checks if the table is already written to disk in which case the
        data is loaded into memory from disk by calling the _read_table
        function. Otherwise the function _scrape_table is called in which
        case the data is scraped from the internet.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Read data from disk if it already exists
        if (not force_download) & self.fpath.exists():
            self._read_table()
    
        # Otherwise, scrape data from WWW
        else:
            await self._scrape_table()


class MetaInfoTableLoader(object):
    """
    Class handling loading meta information
    about MONET2030 indicator data into memory.

    Parameters
    ----------
    indicator_df : pd.DataFrame
        Dataframe containing all MONET2030 indicators
        together with the URLs pointing to the respective
        subpages.

    metainfo_table_path : str
        Path where the meta information table is stored
        or will be written to.
    """
    def __init__(self, indicator_df: pd.DataFrame, metainfo_table_path: str):
        self.indicators = indicator_df
        self.fpath = metainfo_table_path
        self.table = None
        
    async def _scrape_table(self):
        """
        Scrapes the indicator meta information table from
        the internet.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        df_list = []
        counter = 0
        n_indicators = len(self.indicators)
        print("Scraping...")
        # The following loop iterates over every MONET2030 indicator.
        # Specifically, this means that for every indicator the meta
        # information for the corresponding data file(s) is extracted
        # and transformed. For each one, a dataframe containing this
        # meta information is created. These dataframes are appended
        # to the df_list for subsequent concatenation into one single
        # big dataframe.
        for idx, indicator in self.indicators.iterrows():
            counter += 1
            print(f"{counter}/{n_indicators}", end="\r")
        
            # ETL process for specific Monet2030 indicator
            etl_mii = etl.ETL_MonetIndicatorInfo(indicator["hyperlink"])
            await etl_mii.extract()
            etl_mii.transform()
        
            # Augment data
            etl_mii.df["indicator_id"] = indicator["id"]
            etl_mii.df["sdg"] = indicator["sdg"]
            etl_mii.df["topic"] = indicator["topic"]
            etl_mii.df["indicator"] = indicator["indicator"]
            df_list.append(etl_mii.df)
        print("-> done!")
    
        # Concatenate all small dataframes in df_list to one big
        # dataframe
        self.table = pd.concat(df_list, ignore_index=True)

        # Resort columns
        self.table = self.table[["dam_id", "indicator_id", "sdg", "topic", "indicator", "observable", "description", "units", "data_file_url"]]
        self.table.set_index("dam_id", inplace=True)
        
        # Write table to file
        Path("/".join(self.fpath.as_posix().split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        self.table.to_csv(self.fpath)

        # Unset index (if any)
        self.table = self.table.reset_index()
    
    def _read_table(self):
        """
        Reads the indicator meta information table from
        disk, assuming the file exists at the location
        indicated by self.fpath.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
    
        Raises
        ------
        FileNotFoundError
            If the file cannot be found at self.fpath
        """
        if not self.fpath.exists():
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT),
                                    self.fpath)
            
        print("Reading from disk...")
        self.table = pd.read_csv(self.fpath)
        if "index" in self.table.columns:
            self.table.drop("index", axis=1, inplace=True)
        print("-> done!")
    
    async def get_table(self, force_download=False):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Read data from disk if it already exists
        if (not force_download) & self.fpath.exists():    
            self._read_table()

        # Otherwise, scrape data from WWW
        else:
            start = dt.now()
            await self._scrape_table()
            end = dt.now()
            elapsed = end - start
            print(f"Finished after {elapsed.seconds} seconds.")


class DataFileLoader(object):
    """
    Class handling loading into memory and 
    transforming the actual MONET2030 indicator
    data .

    Parameters
    ----------
    metatable : pd.DataFrame
        Dataframe containing metainformation about
        MONET2030 indicators together with the URLs
        pointing to the actual data files.

    raw_data_path : str
        Path to directory in which the raw data files 
        are stored exactly as they were downloaded.
        These files are not processed at all.

    processed_data_path : str
        Path to directory in which the processed
        data files are stored. They are derived
        from the raw data files but are more friendly
        for subsequent data analysis.
    """
    def __init__(self, metatable: pd.DataFrame, raw_data_path: str, processed_data_path: str):
        self.metatable = metatable
        self.raw_fpath = raw_data_path
        self.processed_fpath = processed_data_path
        self.raw_data_list = []
        self.processed_data_list = {"stage1": [], "stage2": []}
        self.log = {"raw": None, "processed": dict()}

    def _compactify

    def _scrape_data(self):
        """
        Scrapes the MONET2030 indicator data from the WWW.
        The raw data is subsequently transformed into a more
        analysis-friendly form. Both, raw and transformed
        data are written to disk.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        print("Scraping...")
        n_rows = len(self.metatable)

        raw_log = {"file_id": [],
                   "file_name": [],
                   "file_hash": [],
                   "dam_id": [],
                   "download_date": [],
                   "download_timestamp": [],
                   }
        processed_s1_log = {"file_id": [],
                            "file_name": [],
                            "file_hash": [],
                            "dam_id": [],
                            "processed_date": [],
                            "processed_timestamp": [],
                            }

        processed_s2_log = {"file_id": [],
                            "file_name": [],
                            "file_hash": [],
                            "metric_id": [],
                            "dam_id": [],
                            "processed_date": [],
                            "processed_timestamp": [],
                            }
        
        for counter, (idx, row) in enumerate(self.metatable.iterrows()):
            print(f"{(counter+1)}/{n_rows}", end="\r")

            # download & process data
            # -----------------------
            etl_df = etl.ETL_DataFile(row)
            etl_df.extract()
            downloaded = dt.now(tz=const.zurich_tz)
            etl_df.transform()
            
            # Augment/enrich processed data for all
            # processing stages
            processed = dict()
            etl_df.processed_data["stage1"]["observable"] = row["observable"]
            etl_df.processed_data["stage1"]["dam_id"] = row["dam_id"]
            processed["stage1"] = dt.now(tz=const.zurich_tz)

            for proc_dict in etl_df.processed_data["stage2"]:
                proc_dict["observable"] = row["observable"]
                proc_dict["dam_id"] = row["dam_id"]
            processed["stage2"] = dt.now(tz=const.zurich_tz)

            
            # ==================
            # write data to file
            # ==================
            # 1) RAW DATA
            # -----------
            # define file name root
            file_name_root = f"m2030ind_damid_{str(row["dam_id"]).zfill(8)}"

            # Write raw data to file
            self.raw_fpath.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(self.raw_fpath / (file_name_root + ".xlsx")) as writer:
                for name, df in etl_df.raw_spreadsheet.items():
                    df.to_excel(writer, sheet_name=name)

            # Make data available
            self.raw_data_list.append(etl_df.raw_spreadsheet)

            # Collect logging info
            raw_hash = aux.xlsx_hasher(etl_df.raw_spreadsheet)
            raw_log["file_id"].append(f"{row["dam_id"]}_r")
            raw_log["file_name"].append(file_name_root+".xlsx")
            raw_log["file_hash"].append(raw_hash)
            raw_log["dam_id"].append(row["dam_id"])
            raw_log["download_date"].append(downloaded.strftime(format="%Y-%m-%d"))
            raw_log["download_timestamp"].append(downloaded.strftime(format="%H:%M:%S"))

            # Write log files
            raw_log_df = pd.DataFrame(raw_log).set_index("file_id")
            raw_log_df.to_csv(const.log_file_raw_data)
            self.log["raw"] = raw_log_df

            # 2) STAGE-1-PROCESSED DATA
            # -------------------------
            # REMARK: It seems very verbose and not adhering to the
            # DRY principle to spell out the saving of each processing
            # stage. However, it would be relatively convoluted and
            # not very readable to put this in a for loop as every
            # processing stage has to be treated slightly differently.
            # For instance, the keys in the key_order list is slightly
            # different.
            stage1_data = etl_df.processed_data["stage1"]
            
            # Serialize the processed data to json string
            key_order = ["dam_id", "observable", "description", "remark", "data"]
            ordered_dict = aux.reorder_keys(stage1_data, key_order)
            serializable_data = {k: aux.serialize_value(v) for k, v in ordered_dict.items()}

            # Make data available
            self.processed_data_list["stage1"].append(ordered_dict)
            
            # Write processed data to json file
            dirpath = self.processed_fpath / "stage_1"
            dirpath.mkdir(parents=True, exist_ok=True)
            with open(dirpath / (file_name_root + ".json"), 'w') as f:
                data_json_str = json.dump(serializable_data, f, indent=2)

            # Collect logging info
            processed_hash = aux.json_hasher(json.dumps(serializable_data))
            processed_s1_log["file_id"].append(f"{row["dam_id"]}_p")
            processed_s1_log["file_name"].append(file_name_root+".json")
            processed_s1_log["file_hash"].append(processed_hash)
            processed_s1_log["dam_id"].append(row["dam_id"])
            processed_s1_log["processed_date"].append(processed["stage1"].strftime(format="%Y-%m-%d"))
            processed_s1_log["processed_timestamp"].append(processed["stage1"].strftime(format="%H:%M:%S"))

            # Write log files
            processed_log_s1_df = pd.DataFrame(processed_s1_log).set_index("file_id")
            processed_log_s1_df.to_csv(const.log_file_processed_s1_data)
            self.log["processed"]["stage1"] = processed_log_s1_df

            # 3) STAGE-2-PROCESSED DATA
            # -------------------------
            stage2_data = etl_df.processed_data["stage2"]
        
            key_order = ["metric_id", "dam_id", "observable", "description", "remark", "data"]

            # Treat every metric individually
            for metric_dict in stage2_data: 
                # Adding the most granular metric id
                metric_dict["metric_id"] = str(metric_dict["dam_id"]) + metric_dict["sub_id"]
                ordered_dict = aux.reorder_keys(metric_dict, key_order)
                serializable_data = {k: aux.serialize_value(v) for k, v in ordered_dict.items()}
    
                # Make data available
                self.processed_data_list["stage2"].append(ordered_dict)
                
                # Write processed data to json file
                dirpath = self.processed_fpath / "stage_2"
                dirpath.mkdir(parents=True, exist_ok=True)
                with open(dirpath / (file_name_root + metric_dict["sub_id"] + ".json"), 'w') as f:
                    data_json_str = json.dump(serializable_data, f, indent=2)

                # Collect logging info
                processed_hash = aux.json_hasher(json.dumps(serializable_data))
                processed_s2_log["file_id"].append(f"{metric_dict["metric_id"]}")
                processed_s2_log["file_name"].append(file_name_root+metric_dict["sub_id"]+".json")
                processed_s2_log["file_hash"].append(processed_hash)
                processed_s2_log["dam_id"].append(row["dam_id"])
                processed_s2_log["metric_id"].append(metric_dict["metric_id"])
                processed_s2_log["processed_date"].append(processed["stage2"].strftime(format="%Y-%m-%d"))
                processed_s2_log["processed_timestamp"].append(processed["stage2"].strftime(format="%H:%M:%S"))
    
            # Write log files
            processed_log_s2_df = pd.DataFrame(processed_s2_log).set_index("file_id")
            processed_log_s2_df.to_csv(const.log_file_processed_s2_data)
            self.log["processed"]["stage2"] = processed_log_s2_df
        
        print("-> done!")

    def _read_data(self):
        """
        Reads both raw and processed data from disk, assuming
        those data files exist at the indicated locations.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        print("Reading raw data from disk...")
        sorted_xlsx_files = sorted([file for file in self.raw_fpath.glob("*.xlsx")])
        self.raw_data_list = [(file.as_posix().split("/")[-1].split(".")[0].split("_")[-1], 
                               pd.read_excel(file, sheet_name=None)
                              ) for file in sorted_xlsx_files
                             ]
        print("-> done!")

        print("Reading processed data from disk...")
        sorted_json_files = sorted([file for file in self.processed_fpath.glob("*.json")])

        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.processed_data_list.append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

    def get_data(self, force_download=False):
        """
        Reads MONET2030 indicator data tables into memory.
    
        Checks if the tables are already written to disk in which case the
        data is loaded into memory from disk by calling the _read_data_tables
        function. Otherwise the function _scrape_indicator_table is called in
        which case the data is scraped from the internet.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Read data from disk if it already exists
        n_files_expected = self.metatable["dam_id"].nunique()
        paths_exist = self.raw_fpath.exists() and self.processed_fpath.exists()
        dirs_not_empty = (len([f for f in self.raw_fpath.glob("*.xlsx")])==n_files_expected)\
                        &(len([f for f in self.processed_fpath.glob("*.json")])==n_files_expected)
        
        if (not force_download) & paths_exist & dirs_not_empty:
            self._read_data()
    
        # Otherwise, scrape data from WWW
        else:
            self._scrape_data()
        
        
def main():
    """
    Main function scraping or reading the MONET2030
    indicator data.
    """
    # -----------------------------------
    # 1) List of all MONET2030 indicators
    # -----------------------------------
    # First, let's get a list of all indicators and their meta information
    # (e.g. the URLs pointing to the indicator-specific subpages). 
    itl = IndicatorTableLoader(const.url_all_monet2030_indicators, 
                               const.indicator_table_path
                              )
    itl.get_table()
    
    # ------------------------------------------------------
    # 2) List of all data files for all MONET2030 indicators
    # ------------------------------------------------------
    # Given a list of all subpages related to the MONET2030 indicators (see Step 1),
    # we can now go a step further and scrape each of these subpages. Doing so we can
    # find yet a new set of URLs that point to the actual indicator-specific data 
    # files. It is the data in these files we are ultimately interested in.
    mitl = MetaInfoTableLoader(itl.table,
                               const.metainfo_table_path
                               )
    mitl.get_table()
        
    # ------------------------------
    # 3) Download all the data files
    # ------------------------------
    # Download the data files listed in the metadata table created in the previous
    # step. The data files are stored in raw as well as processed format.
    dfl = DataFileLoader(mitl.table,
                         const.raw_data_path,
                         const.processed_data_path
                        )
    dfl.get_data()

if __name__ == "__main__":
    main()