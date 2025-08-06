# Stdlib imports
import re
import requests
import zipfile
from pathlib import Path

# 3rd party imports
import pandas as pd

# Local imports
from pywise import wise_consts as const

class WiseLoader(object):
    """
    Methods
    -------
    load() -> pandas.DataFrame
    """
    def __init__(self):
        self.wise_db = dict()
        
    def _download(self) -> None:
        """
        Download the WISE data from the WWW.
        """
        r = requests.get(const.wise_download_url, stream=True)
        return r

    @staticmethod
    def _write2disk(req):
        """
        Write the WISE database (a zip file)
        to disk.
        """
        # Write ZIP file to disk
        
        with open(const.wise_db_zippath, 'wb') as fd:
            for chunk in req.r.iter_content(chunk_size=128):
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

    def _read(self):
        """
        Read the WISE database from disk.
        """
        for sheet in ["Metrics Info", "C Data"]: # The sheets "Content", "CG Data", "Metrics C&CG", "C&CG Code" are not needed.
            self.wise_db[sheet] = pd.read_excel(const.wise_db_fpath, sheet_name=sheet)

    def load(self):
        """
        Load the WISE database into memory.

        If the database is not already on disk,
        the database is downloaded from the web.

        Returns
        -------
        wise_db : pandas.DataFrame
            DataFrame containing the WISE data.
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
        self._read()
        print("-> Done!")
        return self.wise_db