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
        self.processed_data = dict()

    def extract(self):
        """
        Downloads the actual data excel spreadsheets
        from the web. Notice that this data is not
        in data analysis-friendly format. This is 
        fixed in the transform method below.
        """
        self.raw_spreadsheet = pd.read_excel(self.metainfo["data_file_url"], sheet_name=None)

    def _return_worksheets(self):
        """
        Returns a list of work sheets in the spreadsheet
        self.raw_spreadsheet.

        Returns
        -------
        worksheets : List[str]
            List of work sheet names.
        """
        worksheets = list(self.raw_spreadsheet.keys())
        return worksheets

    def _get_table_name(self, table) -> str:
        """
        Extracts the name of the MONET2030
        indicator from the datafile itself.

        The name is located in the top-left
        cell (coordinates 0,0).

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        table_name : str
            Name of the current table/observable
        """
        table_name = table.iloc[0,0]
        return table_name
        
    def _get_table_description(self, table) -> str:
        """
        Extracts the description of the MONET2030
        indicator from the datafile itself.

        The name is located in the cell with
        coordinates (0,1).

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        table_description : str
            Description of the current table/observable
        """
        table_description = table.iloc[1,0]
        return table_description

    def _get_dataframe_and_remarks(self, table, description) -> Tuple[pd.DataFrame, str]:
        """
        Extracts the actual data (as a dataframe)
        and possibly a remark from the table.

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        description : str
            Description of the current table/observable

        Returns
        -------
        df : pandas.DataFrame
            DataFraem containing the actual data for the
            current observable.

        remark : str
            Remarks found at the end of the spreadsheet.
        """
        # Check if there actually is a description
        # or whether it's missing
        if description is np.nan:
            column_headers_row = 2
        else:
            column_headers_row = 3

        # Get all the column names for the current data set
        # They follow after the description and a blank line
        col_names = [v for v in table.iloc[column_headers_row,:].values if (v is not np.nan and len(v.strip())>0)]

        # Extract the data (at this point including additional
        # information and footers)
        df = table.iloc[column_headers_row+1:,:]

        # Now let's put in the column names we extracted
        # above.
        col_rename_dict = dict()
        col_rename_dict[df.columns[0]] = "Year"
        for i in range(1, len(col_names)+1):
            col_rename_dict[df.columns[i]] = col_names[i-1]
        df = df.rename(col_rename_dict, axis=1)
        df = df.set_index("Year")

        # Figure out where the actual data
        # ends and where the footer starts
        for cntr, idx in enumerate(df.index):
            if idx!=idx:
                stop = cntr
                break

        # Use this info to extract the remarks (footer)
        remark = " ".join([str(txt) for txt in df.index[stop:] if txt==txt])

        # Separate the actual data
        df = df.iloc[:stop,:]

        return df, remark

    def _stage1_transformation(self, table: pd.DataFrame) -> Dict:
        """
        Performs the stage 1 transformation of the
        MONET2030 data.

        Parameters
        ----------
        table : pandas.DataFrame
            The tabel in the zeroth worksheet
            of self.raw_spreadsheet

        Returns
        -------
        s1_res : Dict
            Dictionary containing the actual data as a dataframe,
            the description and the remark about the data in that
            dataframe.
        """
        # Get the name
        name = self._get_table_name(table)

        # Get the description
        desc = self._get_table_description(table)

        # Get the actual data table and remarks
        df, remark = self._get_dataframe_and_remarks(table, desc)

        # Give the columns a name
        df.columns.name = name

        # Collect the stage-1-transformation results and
        # return
        s1_res = {"data": df, "description": desc, "remark": remark}
        return s1_res

    def _stage2_transformation(self, obs_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        From a MONET2030 observable dataframe that 
        contains several columns (subobservables), 
        create one dataframe per subobservable/
        column.

        Parameters
        ----------
        obs_df : pandas.DataFrame
            DataFrame containing all the data
            for a specific MONET2030 observable.

        Returns
        -------
        subobs_df_list : List[pandas.DataFrame]
            List of single-column dataframes.
            Each dataframe contains data about
            one specific subobservable.
        """

        subobs_df_list = [] # metric
        ci95_df_list = []   # 95% confidence intervals
    
        # Iterate over all columns and create
        # a separate dataframe for each one
        for col_idx, col in enumerate(obs_df.columns):
            if "confidence interval" in col.lower():
                # Rename the column to integrate the actual observable name
                new_col_name = f"{col} ({obs_df.columns[col_idx-1]})"
                new_df = pd.DataFrame(index = obs_df.index, columns = [new_col_name])
                new_df[new_col_name] = obs_df.iloc[:, col_idx]
                ci95_df_list.append(new_df)
            else:
                subobs_df_list.append(obs_df[[col]])

        return subobs_df_list, ci95_df_list
        
    def transform(self):
        """
        Takes the data original data spreadsheets
        and transforms them into analysis-friendly
        format. This will create single-column pandas
        dataframes - one for each individual atomic
        observable.
        """
        # Get a list of work sheets in the spreadsheet
        # self.raw_spreadsheet
        sheetnames = self._return_worksheets()

        # Extract the actual data table (which is the first sheet)
        table = self.raw_spreadsheet[sheetnames[0]]

        # -------
        # STAGE 1
        # -------
        s1_trafo_results = self._stage1_transformation(table)
        self.processed_data["stage1"] = s1_trafo_results
        
        # -------
        # STAGE 2
        # -------
        observable_df = s1_trafo_results["data"]
        subobservable_df_list, conf_intv_list = self._stage2_transformation(observable_df)

        s2_trafo_results = []
        for subobs_id, subobs in enumerate(subobservable_df_list):
            s2_trafo_results.append({"data": subobs, 
                                     "description": s1_trafo_results["description"], 
                                     "remark": s1_trafo_results["remark"],
                                     "type": "metric",
                                     "sub_id": chr(97+subobs_id)+"_metr"
                                    }
                                   )
            
        for ci_id, ci in enumerate(conf_intv_list):
            s2_trafo_results.append({"data": ci, 
                                     "description": s1_trafo_results["description"], 
                                     "remark": s1_trafo_results["remark"],
                                     "type": "confidence interval",
                                     "sub_id": chr(97+ci_id)+"_ci"
                                    }
                                   )

        self.processed_data["stage2"] = s2_trafo_results
        
        