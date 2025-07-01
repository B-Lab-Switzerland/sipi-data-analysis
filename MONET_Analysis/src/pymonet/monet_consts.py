from pathlib import Path
import pytz

# Project-wide timezone definition
zurich_tz = pytz.timezone("Europe/Zurich")

# File paths
indicator_table_path = Path("../data/metadata/indicator_table.csv")
metainfo_table_path = Path("../data/metadata/monet_datafile_summary_table.csv")
log_file_raw_data = Path("../data/metadata/raw_data_log.csv")
log_file_processed_data = Path("../data/metadata/processed_data_log.csv")
raw_data_dir = Path("../data/raw/")
processed_data_dir = Path("../data/processed/")

# URLs
url_all_monet2030_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html'
    