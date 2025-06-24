# Stdlib imports
import re

# 3rd party importsimport requests
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Local imports
from pymonet import monet_aux as aux

###
# 1) SCRAPE META INFORMATION OF MONET INDICATORS
###
async def parse_dynamic_webpage(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_timeout(5000)
        html = await page.content()
        await browser.close()
        return html


# Create a list of all MONET2030 indicators as well as their mapping to SDGs
def create_monet_indicator_list(html):
    # Filter out relevant parts
    indicators = BeautifulSoup(html).find("tbody").find_all("tr")
    
    # Oragnize the MONET indicators into a DataFrame
    base_url = "https://www.bfs.admin.ch"
    sdgs = [{"SDG": int(indicator.find_all("a")[0]["aria-label"].split(":")[0].split()[1].strip()),
            "Topic": indicator.find_all("a")[0]["aria-label"].split(":")[1].strip(),
            "Indicator": indicator.find_all("a")[1]["aria-label"],
            "Hyperlink": base_url+indicator.find_all("a")[1]["href"].replace("content/",""),
            "Agenda2030_relevant": 1 if len(indicator.find_all("img", {"title": "Agenda 2030: relevant"}))==1 else 0
            } for indicator in indicators]
    
    sdg_df = pd.DataFrame(sdgs)

    # Add a unique identifier to each indicator
    sdg_df["SubtopicID"] = sdg_df.groupby("SDG").cumcount().add(1).astype(str)
    sdg_df["ID"] = sdg_df.apply(lambda x: "MI-" + str(x["SDG"]) + "." + x["SubtopicID"], axis=1)
    sdg_df.drop("SubtopicID", axis=1, inplace=True)
    sdg_df.set_index("ID", inplace=True)
    
    # Return
    return sdg_df

###
# 2) THE CODE BELOW IS TO SCRAPE THE URLS TO THE ACTUAL DATA FILES
###
async def scrape_indicator_info(url):
    html = await parse_dynamic_webpage(url)
    soup = BeautifulSoup(html, 'html.parser')
    ul = soup.find("ul", {"class": "search-results-list", "data-vue-component": "asset-list"})
    if not ul:
        return []
    data_file_elements = ul.find_all("li")
    return data_file_elements

def extract_all_data_files(data_file_elements):
    data_file_info_list = []
    for elm in data_file_elements:
        damid_list = re.findall(r'damid="\d+"', str(elm))
        assert len(damid_list)==1
        damid = re.search(r'\d+', damid_list[0]).group(0)
        desc_string = strip_tags(str(elm.find("div", {"class": "card__title"})))
        data_info = desc_string.split(" - ")
        if len(data_info)==2:
            observable = data_info[0]
            description = data_info[1]
            units = ""
        elif len(data_info)==3:
            observable = data_info[0]
            description = data_info[1]
            units = data_info[2]
            
        file_dict = {"damid": damid, 
                     "Data_url": f"https://dam-api.bfs.admin.ch/hub/api/dam/assets/{damid}/master",
                     "Observable": observable, 
                     "Description": description,
                     "Units": units,
                    }
        data_file_info_list.append(file_dict)
    return pd.DataFrame(data_file_info_list)