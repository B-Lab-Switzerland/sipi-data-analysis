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
        data is loaded into memory from disk by calling the _read_indicator_table.
        Otherwise the function _scrape_indicator_table is called in which
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
    """
    def __init__(self, metatable: pd.DataFrame, raw_data_path: str, processed_data_path: str):
        self.metatable = metatable
        self.raw_fpath = raw_data_path
        self.processed_fpath = processed_data_path
        self.raw_data_list = []
        self.processed_data_list = []

    def _table_reformatter(self, spreadsheet: Dict[str, pd.DataFrame]) -> Dict:
        """
        """
        sheetnames = list(spreadsheet.keys())
    
        table = spreadsheet[sheetnames[0]]
        name = table.iloc[0,0]
        desc = table.iloc[1,0]
        if desc is np.nan:
            column_headers_row = 2
        else:
            column_headers_row = 3
        
        col_names = [v for v in table.iloc[column_headers_row,:].values if (v is not np.nan and len(v.strip())>0)]
        
        df = table.iloc[column_headers_row+1:,:]
        
        col_rename_dict = dict()
        col_rename_dict[df.columns[0]] = "Year"
        for i in range(1, len(col_names)+1):
            col_rename_dict[df.columns[i]] = col_names[i-1]
        df = df.rename(col_rename_dict, axis=1)
        df = df.set_index("Year")
        
        for cntr, idx in enumerate(df.index):
            if idx!=idx:
                stop = cntr
                break
        
        remark = " ".join([str(txt) for txt in df.index[stop:] if txt==txt])
        df = df.iloc[:stop,:]
        df.columns.name = name
        df
    
        return {"table": df,
                "desc": desc,
                "remark": remark}
    
    def _pull_raw_data(self):
        for idx, row in self.metatable.iterrows():
            doc = pd.read_excel(row["Data_url"], sheet_name=None)

            file_name = f"m2030ind_damid_{row["damid"]}.xlsx"
            with pd.ExcelWriter(self.raw_fpath / file_name) as writer: 
                for name, data in doc.items():
                    data.to_excel(writer, sheet_name=name)
            
            self.raw_data_list.append(doc)
            

    def reformat(self):
        for doc in self.raw_data_list:
            proc_doc = self._table_reformatter(doc)
            self.processed_data_list.append(proc_doc)
    
    def get_raw_data(self):
        pass
        

def main():
    """
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
    dfl = DataFileLoader(mitl.table,
                         const.raw_data_path,
                         const.processed_data_path
                        )

if __name__ == "__main__":
    main()
    