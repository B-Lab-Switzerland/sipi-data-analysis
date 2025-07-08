from pathlib import Path
import pytz

# Project-wide timezone definition
zurich_tz = pytz.timezone("Europe/Zurich")

# File paths
indicator_table_path = Path("../data/metadata/indicator_table.csv")
metainfo_table_path = Path("../data/metadata/monet_datafile_summary_table.csv")
log_file_raw_data = Path("../data/metadata/raw_data_log.csv")
log_file_processed_s1_data = Path("../data/metadata/processed_s1_data_log.csv")
log_file_processed_s2_data = Path("../data/metadata/processed_s2_data_log.csv")
raw_data_dir = Path("../data/raw/")
processed_data_dir = Path("../data/processed/")

non_redundant_obs_file = Path("../results/non_redundant_observables.csv")
all_corrmat_file = Path("../results/corrmat_all.pdf")
pruned_corrmat_file = Path("../results/corrmat_pruned.pdf")
pruned_metrics_file = Path("../results/pruned_metrics.xlsx")

compact_metrics_filename = "monet2030_metrics.csv" 
compact_cis_filename = "monet2030_confintervals.csv"

# URLs
url_all_monet2030_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html'
    