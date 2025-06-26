# Std lib imports
import errno
import os
from datetime import datetime as dt
from typing import Dict

# 3rd party imports
import pandas as pd
import numpy as np

# Local imports
from pymonet import monet_etl as etl

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
        print("-> done!")
    
    async def get_table(self):
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
        if self.fpath.exists():
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
            etl_mii = etl.ETL_MonetIndicatorInfo(indicator["Hyperlink"])
            await etl_mii.extract()
            etl_mii.transform()
        
            # Augment data
            etl_mii.df["Indicator"] = indicator["Indicator"]
            etl_mii.df["SDG"] = indicator["SDG"]
            etl_mii.df["Topic"] = indicator["Topic"]
            df_list.append(etl_mii.df)
        print("-> done!")
    
        # Concatenate all small dataframes in df_list to one big
        # dataframe
        self.table = pd.concat(df_list, ignore_index=True)
    
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
        print("-> done!")
    
    async def get_table(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Read data from disk if it already exists
        if self.fpath.exists():    
            self._read_table()

        # Otherwise, scrape data from WWW
        else:
            start = dt.now()
            await self._scrape_table()
            end = dt.now()
            elapsed = end - start
            print(f"Finished after {elapsed.seconds} seconds.")
        
            # Resort columns
            self.table = self.table[["SDG", "Topic", "Indicator", "Observable", "Description", "Units", "damid", "Data_url"]]
            

class DataFileLoader(object):
    """
    Class handling loading the actual MONET2030
    indicator data into memory.

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
        self.processed_data_list = []

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
        for counter, (idx, row) in enumerate(self.metatable.iterrows()):
            print(f"{(counter+1)/n_rows}")
            etl_df = etl.ETL_DataFile(row)
            etl_df.extract()
            etl_df.transform()

            # Make data available
            self.raw_data_list.append(etl_df.raw_spreadsheet)
            self.processed_data_list.append(etl_df.processed_data)
            
            # Prepare writing to file
            file_name_root = f"m2030ind_damid_{row["damid"]}"

            # Write raw data to file
            with pd.ExcelWriter(self.raw_fpath / (file_name_root + ".xlsx")) as writer: 
                etl_df.raw_spreadsheet.to_excel(writer, sheet_name=name)

            # Write processed data to file
            etl_df.processed_data.to_csv(self.processed_fpath / (file_name_root + ".csv"))
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
        self.raw_data_list = [pd.read_excel(file, sheet_name=None) for file in self.raw_fpath.glob("*.xlsx")]
        print("-> done!")

        print("Reading processed data from disk...")
        self.processed_data_list = [pd.read_csv(file) for file in self.processed_fpath.glob("*.xlsx")]
        print("-> done!")

    def get_data(self):
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
        if self.raw_fpath.exists() and self.processed_fpath.exists():
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