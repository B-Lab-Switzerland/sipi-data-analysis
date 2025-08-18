# Stdlib imports
import re
import requests
import zipfile
from pathlib import Path

# 3rd party imports
import pandas as pd

# Local imports
from pywise import wise_consts as const
from pywise import wise_aux as aux

class WiseLoader(object):
    """
    Methods
    -------
    load() -> pandas.DataFrame
    """
    def __init__(self):
        self.wise_db = dict()
        self.metatable = None
        
    def _download(self) -> None:
        """
        Download the WISE data from the WWW.

        Returns
        -------
        req : requests.models.Response
            Response from requests.get
        """
        r = requests.get(const.wise_download_url, stream=True)
        return r

    @staticmethod
    def _write2disk(req):
        """
        Write the WISE database (a zip file)
        to disk.

        Parameters
        ----------
        req : requests.models.Response
            A response from a call to requests.get
        """
        # Write ZIP file to disk
        
        with open(const.wise_db_zippath, 'wb') as fd:
            for chunk in req.iter_content(chunk_size=128):
                fd.write(chunk)

    @staticmethod
    def _unzip():
        """
        Unzip the WISE database zip file.

        The zipfile at location const.wise_db_zippath
        is unzipped, upon which an Excel spreadsheet
        at location const.wise_db_fpath is created.
        """
        with zipfile.ZipFile(const.wise_db_zippath, 'r') as zip_ref:
            zip_ref.extractall(const.raw_dir)

    def _read(self, fpath: Path):
        """
        Read the WISE database from disk.
        """
        sheet_list = ["Metrics Info", "C Data"]
        wise_db = pd.read_excel(fpath, 
                                sheet_name=sheet_list,
                                engine="openpyxl",
                                parse_dates=False
                               )
        return wise_db

    def _create_metatable(self):
        """
        """
        loaded_from_disk = False

        # Load the meta information as provided by WISE
        try:
            metatable = pd.read_csv(const.wise_metatable_fname)
            if "Unnamed: 0" in metatable.columns:
                metatable = metatable.drop("Unnamed: 0", axis=1)

            
            loaded_from_disk = True
        except FileNotFoundError:
            metatable = self.wise_db["Metrics Info"].drop("Unnamed: 0", axis=1)
            metatable = aux.standardize_column_names(metatable)

        # Standardize column headers
        metatable = aux.standardize_column_names(metatable)
        metatable = metatable.set_index("acronym")

        # Add in capital information
        if "capital - primary" not in metatable.columns:
            capitals_map = pd.read_csv(const.capmap_path)
            if "Unnamed: 0" in capitals_map.columns:
                capitals_map = capitals_map.drop("Unnamed: 0", axis=1)
                
            capitals_map = aux.standardize_column_names(capitals_map)
            capitals_map = capitals_map[["acronym","capital - primary"]]
            capitals_map = capitals_map.set_index("acronym")
            
            metatable = metatable.join(capitals_map)
            
        # Align meta table header and row names
        metatable = metatable.rename({"metric_full_name": "metric_name"}, axis=1)
        metatable.index = [ri.lower() for ri in metatable.index]

        metatable.index.name="acronym"

        if not loaded_from_disk:
            # Save to file
            dirpath = const.wise_metatable_fname.parent
            dirpath.mkdir(parents=True, exist_ok=True)
            metatable.to_csv(const.wise_metatable_fname) 
        
        self.metatable = metatable

    def load_single_country(self, country_iso3: str):
        """
        """
        file_path = const.single_country_wise_db_fpath(country_iso3)
        if not (file_path).exists():
            print(f"File {file_path} not found.")
            if not (const.wise_db_fpath).exists():
                print(f"File {const.wise_db_fpath} also not yet available --> Downloading...")
                req = self._download()
                self._write2disk(req)
                self._unzip()
                print("-> Download complete.")
                        
            print("Reading database into memory...")
            full_wise_db = self._read(const.wise_db_fpath)
            print("-> Reading complete.")

            print(f"Keeping only data for {country_iso3}...")
            metrics_data = full_wise_db["C Data"]
            
            single_country_wise_db = dict()
            single_country_wise_db["C Data"] = metrics_data[metrics_data["ISO3"]==country_iso3]
            single_country_wise_db["Metrics Info" ] = full_wise_db["Metrics Info"]
            print("-> Complete.")

            print("Saving country-specific file to disk...")
            dirpath = file_path.parent
            dirpath.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(file_path) as writer:
                for name, df in single_country_wise_db.items():
                    df.to_excel(writer, sheet_name=name)
            print("-> Saving complete.")
            
        else:
            print("Data already available. No download required.")
            single_country_wise_db = self._read(file_path)
        
        print("-> Done!")
        return single_country_wise_db

    def load_all(self):
        """
        """
        if not (const.wise_db_fpath).exists():
            print("File not yet available --> Downloading...")
            req = self._download()
            self._write2disk(req)
            self._unzip()
            print("-> Download complete.")
        else:
            print("Data already available. No download required.")

        print("Reading database into memory...")
        self._read(const.wise_db_fpath)
        
        print("-> Done!")
        return wise_db

    def load(self, country_iso3: str|None = None):
        """
        Load the WISE database into memory.

        If the database is not already on disk,
        the database is downloaded from the web.

        Optional Parameters
        -------------------
        country_iso3 : str
            ISO3 country code. If defined, only
            data for this country will be loaded
            into memory.

        Returns
        -------
        wise_db : pandas.DataFrame
            DataFrame containing the WISE data.
        """
        if country_iso3:
            wise_db = self.load_single_country(country_iso3)
        else:
            wise_db = self.load_all()

        self.wise_db = wise_db
        print("Creating meta information table...")
        self._create_metatable()

        return wise_db
        