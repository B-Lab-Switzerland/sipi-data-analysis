# Stdlib imports
import requests
import json
from io import StringIO
from html.parser import HTMLParser
from collections import OrderedDict
import hashlib
from typing import Dict, List

# 3rd party imports
import pandas as pd

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    """
    Strips HTML tags from a string.
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def page_exists(url):
    """
    Checks if the webpages pointed to by an URL
    actually exists.
    """
    response = requests.head(url, allow_redirects=True, timeout=10)
    return response.status_code == 200

def serialize_value(val):
    """
    Serialized an input if and only if it is of type pd.DataFrame.
    Otherwise it leaves the input unchanged.
    """
    if isinstance(val, pd.DataFrame):
        return {
            '__type__': 'DataFrame',
            'data': val.to_json(orient='split')
        }
    return val

def deserialize_value(val):
    """
    Deserializes the input value if it is of type "dict" and
    if its __type__ attribute equals 'DataFrame'. Otherwise
    leaves the input value unchanged.
    """
    if isinstance(val, dict) and val.get('__type__') == 'DataFrame':
        return pd.read_json(StringIO(val['data']), orient='split')
    return val


def json_hasher(jsonstr: str):
    """
    Converts a jsonstring into a SHA256 hash.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(jsonstr.encode('utf-8'))
    return hash_obj.hexdigest()

def xlsx_hasher(xlsx_dict: Dict):
    """
    Converts an excel spreasheet (in the form of a 
    dictionary as resulting from pd.read_excel) into
    a SHA256 hash.
    """
    hash_obj = hashlib.sha256()

    # Sort sheet names to ensure consistent order
    for sheet_name, df in xlsx_dict.items():
        sheet_content = df.to_csv(index=False)
        # Update hash with sheet name to differentiate sheets
        hash_obj.update(sheet_name.encode('utf-8'))
        # Update hash with sheet content
        hash_obj.update(sheet_content.encode('utf-8'))

    # Return final hex digest
    return hash_obj.hexdigest()

def json_dump(dirpath, file, jsonstr):
    """
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / file, 'w') as f:
        data_json_str = json.dump(jsonstr, f, indent=2)