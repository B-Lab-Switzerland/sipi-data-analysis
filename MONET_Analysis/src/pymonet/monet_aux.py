# Stdlib imports
import requests
from io import StringIO
from html.parser import HTMLParser
from collections import OrderedDict

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

def reorder_keys(d, key_order):
    """
    Reorders a dictionary d according to key_order and returns
    the result as an OrderedDict.
    """
    return OrderedDict((k, d[k]) for k in key_order if k in d)