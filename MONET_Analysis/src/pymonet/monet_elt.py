# Stdlib imports
import re
from abc import ABC
from typing import List, Tuple, Dict

# 3rd party importsimport requests
# == Web Scraping
import asyncio
import bs4
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# == Data Analysis
import pandas as pd
import numpy as np

# Local imports
from pymonet import monet_aux as aux
from pymonet import monet_load as load


class ELT(ABC):
    """
    Abstract base class providing functionality for
    data extraction, loading, and transformation (ELT).

    This abstract base class unifies the interface for
    the ELT process of different MONET2030 related
    webpages. These processes mainly differ in the
    transformation step which is why the transform
    method is an abstractmethod defined for each 
    child class separately. Each child class corresponds
    to the ELT process of one of the specific MONET2030
    webpages.
    """
    def __init__(self, url: str):
        self.url: str = url
        self.soup: bs4.BeautifulSoup = None
        self.df = None  # type = pandas.DataFrame

    async def extract(self):
        """
        Reads HTML code from webpage.
    
        Uses playwright to read HTML code into python from
        webpage. In contrast to using BeautifulSoup only,
        this approach allows for reading dynamically
        rendered, javascript including webpages. Therefore,
        it is a more flexible approach able to handle a wider
        variety of webpages. However, this comes at the cost
        of slightly increased code complexity as it requires
        the introduction of asynchronous functions (asyncio).
    
        Moreover, playwright allows cross-browser support and
        does not require manual installation of browser drivers
        as selenium would.
    
        Parameters
        ----------
        url : str
            URL of the webpage to read in.
    
        Returns
        -------
        html : str
            HTML string of the webpage pointed to by url.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(self.url)
            await page.wait_for_timeout(5000)
            html = await page.content()
            await browser.close()
            self.soup = BeautifulSoup(html, 'html.parser')

    def transform(self):
        pass


class elt_MonetIndicatorList(ELT):
    """
    Scrapes a list of all MONET2030 indicators from the web.
    """
    def transform(self):
        """
        Create a dataframe of all MONET2030 indicators as well as their
        mapping to SDGs.
        """
        # Filter out relevant parts
        indicators = self.soup.find("tbody").find_all("tr")
        
        # Oragnize the MONET indicators into a DataFrame
        base_url = "https://www.bfs.admin.ch"
        sdgs = [{"sdg": int(indicator.find_all("a")[0]["aria-label"].split(":")[0].split()[1].strip()),
                 "topic": indicator.find_all("a")[0]["aria-label"].split(":")[1].strip(),
                 "indicator": indicator.find_all("a")[1]["aria-label"],
                 "hyperlink": base_url+indicator.find_all("a")[1]["href"].replace("content/",""),
                 "agenda2030_relevant": 1 if len(indicator.find_all("img", {"title": "Agenda 2030: relevant"}))==1 else 0
                } for indicator in indicators]
        
        sdg_df = pd.DataFrame(sdgs)
    
        # Add a unique identifier to each indicator
        sdg_df["subtopic_id"] = sdg_df.groupby("sdg").cumcount().add(1).astype(str)
        sdg_df["id"] = sdg_df.apply(lambda x: "monet-" + str(x["sdg"]) + "." + x["subtopic_id"], axis=1)
        sdg_df.drop("subtopic_id", axis=1, inplace=True)
        sdg_df.set_index("id", inplace=True)
        
        self.df = sdg_df


class elt_MonetIndicatorInfo(ELT):
    """
    Scrapes information about a specific MONET2030 indicator
    as the urls pointing to all data files related to that
    indicator or the description of it.
    """
    def _find_data_elements(self) -> List[bs4.element.Tag]: 
        """
        Finds HTML for each data file.

        The data files are presented on the webpage as unordered
        list items. Therefore, this function extracts the HTML
        of the right/relevant list items and returns those HTML
        codes as a list.

        Parameters
        ----------
        html_soup : bs4.BeautifulSoup
            HTML code of the webpage specific to a given MONET2030
            indicator.

        Returns
        -------
        data_file_htmls : List[bs4.element.Tag]
            A list of HTML codes of the list items containing information
            about the data files related to the indicator in question.
        """
        # Extract the relevant unordered list tag from the HTML soup
        ul = self.soup.find("ul", {"class": "search-results-list", "data-vue-component": "asset-list"})

        # If we cannot find an unordered list in the HTML soup
        # at all, return an empty list
        if not ul:
            return []

        # Inside this unordered list tag, find all individual list items
        # Each list item refers to a data file that belongs to the MONET2030
        # indicator currently looked at.
        data_file_htmls = ul.find_all("li")
        return data_file_htmls

    def transform(self):
        """
        Creates a dataframe with information (columns)
        for all the data files (rows) that belong to
        the current MONET2030 indicator of interest.
        """
        # Get all the information about all data files
        # for the current indicator
        data_files_info = self._find_data_elements()

        # Iterate over all data file information blocks
        data_file_info_list = []  # create a list container
        for info in data_files_info:
            # Each data file for every indicator has an ID, referred to
            # as the "damid". This is one of the most important pieces
            # of information to scrape as the URLs to all data files
            # have exactly the same format. The only piece that varies in
            # these URLs is the damid. In other words, if you know the damid
            # of a specific data file, downloading it is trivial.
            damid_list = re.findall(r'damid="\d+"', str(info))

            # Make sure that per data file there is a unique damid.
            # If that's not the case, something went wrong.
            if len(damid_list)==0:
                raise ValueError("No dam_id found data file.")
            elif len(damid_list)>1:
                raise ValueError("More than one dam_id found for single data file.")
                
            damid = re.search(r'\d+', damid_list[0]).group(0)

            # Extract further information for the current indicator as well as
            # additional information about the distinction between the multiple
            # data files if there is more than one.
            desc_string = aux.strip_tags(str(info.find("div", {"class": "card__title"})))
            data_info = desc_string.split(" - ")
            if len(data_info)==2:
                observable = data_info[0]
                description = data_info[1]
                units = ""
            elif len(data_info)==3:
                observable = data_info[0]
                description = data_info[1]
                units = data_info[2]

            # Arrange all the scraped information into a dictionary for each 
            # file ...
            file_dict = {"dam_id": damid, 
                         "data_file_url": f"https://dam-api.bfs.admin.ch/hub/api/dam/assets/{damid}/master",
                         "observable": observable, 
                         "description": description,
                         "units": units,
                        }
            # ... and append this dictionary to the previously defined
            # container list
            data_file_info_list.append(file_dict)

        # At this point, the list data_file_info_list containes a dictionary
        # for each data file related to the current indicator in question.
        # Transform this list into a pandas.DataFrame.
        self.df = pd.DataFrame(data_file_info_list)
        
class elt_DataFile(object):
    """
    Scrapes the data files for a given observable 
    related to a MONET2030 indicators and processes
    the downloaded raw files into a useable format.

    The downloaded data comes in the form of excel
    spreadsheets. These spreadsheets might contain
    more than one worksheet and the tables are not in a
    useful shape as the contain metainformation such as
    descriptions of the observables or remarks about
    the data gathering. This is fixed and cleaned up
    in the transform method of this class in order
    to produce proper data files (data frames). To avoid
    losing the meta information the data is collected
    in dictionaries that are stored as json files.

    Methods
    -------
    self.extract() : 
        Extracts the data from the web and 
    """
    def __init__(self, metainfo: pd.Series):
        self.metainfo = metainfo
        self.raw_spreadsheet = None

    def extract(self):
        """
        Downloads the actual data excel spreadsheets
        from the web. Notice that this data is not
        in data analysis-friendly format. This is 
        fixed in the transform method below.
        """
        self.raw_spreadsheet = pd.read_excel(self.metainfo["data_file_url"], sheet_name=None)

    
        
        