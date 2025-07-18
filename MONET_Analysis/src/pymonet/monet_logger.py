 # Std lib imports
from typing import Dict, List, Tuple
from pathlib import Path

# 3rd party imports
import pandas as pd

class MonetLogger(object):
    """
    """
    def __init__(self, dict_keys):
        self.dict_keys = dict_keys
        self.log_dict = {k: [] for k in dict_keys}

    def update(self, kv_dict):
        """
        """
        assert set(kv_dict.keys()) == set(self.dict_keys)
        for k, v in kv_dict.items():
            self.log_dict[k].append(v)
        
    def write(self, outfile, index_key=None):
        """
        """
        log_df = pd.DataFrame(self.log_dict)
        if index_key:
            log_df = log_df.set_index("file_id")
        
        log_df.to_csv(outfile)