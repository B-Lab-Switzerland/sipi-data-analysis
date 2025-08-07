# Std lib imports
import errno
import os
import json
from collections import OrderedDict
from datetime import datetime as dt
from typing import Dict, List, Tuple
from pathlib import Path

# 3rd party imports
import pandas as pd
import numpy as np

# Local imports
from pymonet import monet_elt as elt
from pymonet import monet_aux as aux
from pymonet import monet_consts as const

class KeyIndicatorLoader(object):
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
    def __init__(self, key_indicator_url: str, key_indicator_table_path: str):
        self.url = key_indicator_url
        self.fpath = key_indicator_table_path
        self.table = None
        
    #async def _scrape_table(self):
    def _scrape_table(self):
        """
        Scrapes the key indicator table from the WWW.
    
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        raise NotImplementedError("Scraping the key indicator list from the WWW is currently not yet supported.")

    def _read_table(self) -> pd.DataFrame:
        """
        Reads MONET2030 key indicator tabel from disk, assuming 
        the file exists at the location indicated by self.fpath.

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
        self.table = pd.read_csv(self.fpath).set_index("id")
        if "index" in self.table.columns:
            self.table.drop("index", axis=1, inplace=True)
        print("-> done!")
    
    def get_table(self, force_download=False):
        """
        Reads MONET2030 key indicator table into memory.
    
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
            # Once fully implemented, use "await self._scrape_table()"
            # instead of the following code line
            self._scrape_table()

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
    def __init__(self, 
                 capmap: pd.DataFrame,
                 key_indicators_df: pd.DataFrame,
                 indicator_table_url: str, 
                 indicator_table_path: str
                ):
        self.url = indicator_table_url
        self.fpath = indicator_table_path
        self.key_indicator_ids = [kid for kid in key_indicators_df.index]
        self.capitals_map = capmap
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
        # elt process for Monet2030 indicator list
        elt_mil = elt.elt_MonetIndicatorList(self.url)
        await elt_mil.extract()
        elt_mil.transform()
        print("-> done!")

        self.table = elt_mil.df

        # Add "is_key" column
        self.table["is_key"] = False
        self.table.loc[self.table.index.isin(self.key_indicator_ids), "is_key"] = True

        # Join in capitals information (here we join on the index
        # as per the default of the join method).
        self.table = self.table.join(self.capitals_map, how="outer")
        
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
        
            # elt process for specific Monet2030 indicator
            elt_mii = elt.elt_MonetIndicatorInfo(indicator["hyperlink"])
            await elt_mii.extract()
            elt_mii.transform()
        
            # Augment data
            elt_mii.df["indicator_id"] = indicator["id"]
            elt_mii.df["sdg"] = indicator["sdg"]
            elt_mii.df["topic"] = indicator["topic"]
            elt_mii.df["indicator"] = indicator["indicator"]
            elt_mii.df["is_key"] = indicator["is_key"]
            elt_mii.df["capital - primary"] = indicator["capital - primary"]
            elt_mii.df["capital - secondary"] = indicator["capital - secondary"]
            df_list.append(elt_mii.df)
        print("-> done!")
    
        # Concatenate all small dataframes in df_list to one big
        # dataframe
        self.table = pd.concat(df_list, ignore_index=True)

        # Resort columns
        self.table = self.table[["dam_id",
                                 "indicator_id",
                                 "sdg",
                                 "topic",
                                 "indicator",
                                 "observable",
                                 "description",
                                 "units",
                                 "is_key",
                                 "capital - primary",
                                 "capital - secondary",
                                 "data_file_url"
                                ]]
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
    def __init__(self, metatable: pd.DataFrame, raw_data_path: str):
        self.metatable = metatable
        self.raw_fpath = raw_data_path
        self.raw_data_list = []
        self.log = dict()

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
        
        for counter, (idx, row) in enumerate(self.metatable.iterrows()):
            print(f"{(counter+1)}/{n_rows}", end="\r")

            # download data
            # ------------
            elt_df = elt.elt_DataFile(row)
            elt_df.extract()
            downloaded = dt.now(tz=const.zurich_tz)
            elt_df.transform()
            
            # ==================
            # write data to file
            # ==================
            # define file name root
            file_name_root = f"m2030ind_damid_{str(row["dam_id"]).zfill(8)}"

            # Write raw data to file
            self.raw_fpath.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(self.raw_fpath / (file_name_root + ".xlsx")) as writer:
                for name, df in elt_df.raw_spreadsheet.items():
                    df.to_excel(writer, sheet_name=name)

            # Make data available
            self.raw_data_list.append(elt_df.raw_spreadsheet)

            # Collect logging info
            raw_hash = aux.xlsx_hasher(elt_df.raw_spreadsheet)
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
        # Raw data
        # --------
        print("Reading raw data from disk...")
        sorted_xlsx_files = sorted([file for file in self.raw_fpath.glob("*.xlsx")])
        self.raw_data_list = []
        
        for file in sorted_xlsx_files:
            damid = file.as_posix().split("/")[-1].split(".")[0].split("_")[-1]
            xlsx = pd.read_excel(file, sheet_name=None)
            xlsx = {sheetname: df.drop("Unnamed: 0", axis=1, errors="ignore") for (sheetname, df) in xlsx.items()}
            self.raw_data_list.append((damid, xlsx))

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
        paths_exist = self.raw_fpath.exists()
        print(f"paths_exist: {paths_exist}")
        dirs_not_empty = (len([f for f in self.raw_fpath.glob("*.xlsx")])==n_files_expected)
        print(f"dirs_not_empty: {dirs_not_empty}")
        if (not force_download) & paths_exist & dirs_not_empty:
            self._read_data()
    
        # Otherwise, scrape data from WWW
        else:
            self._scrape_data()
        
class MonetLoader(object):
    """
    Loads data from MONET2030 database
    into memory and stores it on disk.

    This class provides the functionality
    to load the MONET2030 database through
    a single user interface provided by the
    method "load". The code automatically 
    checks which files are already available
    on disk and which ones need to be scraped
    from the WWW.

    Parameters
    ----------
    None

    Attributes
    ----------
    indicators_metatable : pandas.DataFrame
        Table listing all high-level MONET2030
        indicators together with additional 
        information.

    observables_metatble : pandas.DataFrame
        Table listing all MONET2030 observables
        (i.e. subindicators) together with
        additional information.

    metatble : pandas.DataFrame
        Table listing all MONET2030 observables
        together with combined additional meta
        information from both indicators_metatable
        and observables_metatble.

    key_indicators_df : pandas.DataFrame
        List of MONET2030 key indicators
        
    capitals_map : pandas.DataFrame
        A mapping of all MONET2030 indicators
        to the four capitals "Social", "Human",
        "Natural", and "Economic".

    Methods
    -------
    load() -> List[Tuple[str, Dict]]
    """
    def __init__(self):
        self.key_indicators_table = None
        self.indicators_metatable = None
        self.observables_metatable = None
        self.metatable = None
        self.capitals_map = self._load_capmap()
        
        # ================================================ #
        # **Important remark:**                            #
        # The capitals mapping file (corresponding to      #
        # const.capmap_path) is a manually created file.   #
        # If it does no longer contain the same ids as     #
        # self.indicators_metatable, it has to be recreated #
        # manually.                                        #
        # ================================================ #
        
    def _load_capmap(self) -> pd.DataFrame:
        """
        Load capitals mapping into memory.

        The capitals map maps the different
        indicators to the corresponding capital
        (social, human, natural, or economic).

        Returns
        -------
        capmap : pandas.DataFrame
            A dataframe listing the indicators
            and the corresponding primary and
            secondary capitals they map to.
        """
        capmap = pd.read_csv(const.capmap_path).set_index("id") 
        return capmap

    def _scrape_keyindicators(self) -> pd.DataFrame:
        """
        List of official MONET2030 key indicators.

        The key indicators are an officially published
        list comprising a subset of MONET2030 indicators
        that are considered particularly important.

        Returns
        -------
        keyinds : pandas.DataFrame
            A dataframe listing the key indicators.
        """
        key = KeyIndicatorLoader(const.url_monet2030_key_indicators,
                                 const.key_indicators_fpath
                                ) 
        key.get_table()
        key_indicators_table = key.table
        return key_indicators_table
        
    async def _scrape_indicators_metatable(self) -> pd.DataFrame:
        """
        List of all MONET2030 indicators
        
        First, let's get a list of all indicators
        and their meta information (e.g. the URLs
        pointing to the indicator-specific subpages). 

        Returns
        -------
        indicators_metatable : pandas.DataFrame
            A list of all indicators (as pandas.DataFrame)
            and their meta information (e.g. the URLs
            pointing to the indicator-specific subpages)
        """
        itl = IndicatorTableLoader(self.capitals_map,
                                   self.key_indicators_table,
                                   const.url_all_monet2030_indicators, 
                                   const.indicator_table_path,
                                  )
        await itl.get_table()
        indicators_metatable = itl.table
        return indicators_metatable

    async def _scrape_observables_metatable(self) -> pd.DataFrame:
        """
        List of all data files for all MONET2030 indicators

        Given a list of all subpages related to the MONET2030
        indicators (see Step 1), we can now go a step further
        and scrape each of these subpages. Doing so we can
        find yet a new set of URLs that point to the actual
        indicator-specific data files. It is the data in these
        files we are ultimately interested in.

        Returns
        -------
        observables_metatable : pandas.DataFrame
            A list of all observables (as pandas.DataFrame)
            and their meta information (e.g. the URLs
            pointing to the individual data files).
        """
        mitl = MetaInfoTableLoader(self.indicators_metatable,
                                   const.metainfo_table_path
                                  )
        await mitl.get_table()
        observables_metatable = mitl.table
        return observables_metatable

    def _scrape_datafiles(self) -> List[Tuple[str, Dict]]:
        """
        Download all the data files.

        Download the data files listed in
        the metadata table created in the
        previous step. The data files are
        stored in raw as well as processed
        format.

        Returns
        -------
        raw_data : List[Tuple[str, Dict]]
            Contains the raw MONET2030 data as downloaded
            from the www. The tuples are of the format
            (dam_id, excel_spreadsheet), where the excel
            spreadsheet is a dictionary with (sheet_name,
            data table) as key-value pairs.
        """
        dfl = DataFileLoader(self.observables_metatable,
                             const.raw_dir,
                             )
        dfl.get_data()
        raw_data = dfl.raw_data_list

        return raw_data

    def _create_full_meta_table(self):
        """
        Create a table with all available
        meta information on observables level.
        """
        # The full meta table is just the join
        # of the observables_metatable and
        # indicators_metatable
        full_meta_table = self.observables_metatable.merge(self.indicators_metatable, 
                                                           left_on="indicator_id",
                                                           right_on="id"
                                                          )

        # Clean up repeated columns
        # -- 1) remove all duplicated columns resulting from join, i.e.
        #       those with suffix "_y"
        repeated_cols = [c for c in full_meta_table.columns if c.endswith("_y")]
        full_meta_table.drop(repeated_cols, axis=1, inplace=True)

        # -- 2) strip away the "_x" suffix wherever necessary
        full_meta_table.columns = [c.replace("_x","") for c in full_meta_table.columns]

        # Add "is_key" column
        full_meta_table["is_key"] = False
        full_meta_table.loc[full_meta_table["indicator_id"].isin(self.key_indicators_df["id"]), "is_key"] = True
        
        self.metatable = full_meta_table
        
    async def load(self) -> List[Tuple[str, Dict]]:
        """
        Scrapes all the web data including meta data
        about indicators and sub-indicators ("observables")
        as well as the actual data.

        Returns
        -------
        raw_data : List[Tuple[str, Dict]]
            Contains the raw MONET2030 data as downloaded
            from the www. The tuples are of the format
            (dam_id, excel_spreadsheet), where the excel
            spreadsheet is a dictionary with (sheet_name,
            data table) as key-value pairs.
        """
        # Step 1
        print("Getting key indicators table...")
        self.key_indicators_table = self._scrape_keyindicators()
        
        # Step 2
        print("Getting indicator information...")
        self.indicators_metatable = await self._scrape_indicators_metatable()
        
        # Step 3
        print("Getting observable information...")
        self.observables_metatable = await self._scrape_observables_metatable()
        
        # Step 4
        print("Getting data files..")
        raw_data = self._scrape_datafiles()
        
        # Return
        return raw_data

def main():
    """
    Main function to execute the web scraper.
    """
    s = MonetLoader()
    s.load()

if __name__ == "__main__":
    scrape_data()