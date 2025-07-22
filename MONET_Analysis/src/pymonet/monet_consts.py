from pathlib import Path
import pytz

# Project-wide timezone definition
zurich_tz = pytz.timezone("Europe/Zurich")

# File paths
# ==========
# Directories
results_dir = Path("../results")
meta_dir = Path("../data/metadata")
raw_dir = Path("../data/raw")
processed_dir = Path("../data/processed")
log_dir = Path("../data/logs")

# Meta files
capmap_path = meta_dir / "capitals_map.csv"
trend_directions = meta_dir / "trend_directions.csv"
key_indicators = meta_dir / "key_indicators.csv"
indicator_table_path = meta_dir / "indicator_table.csv"
metainfo_table_path = meta_dir / "monet_datafile_summary_table.csv"

# Log files
log_file_raw_data = log_dir / "raw_data_log.csv"
log_file_processed_s1_data = log_dir / "processed_s1_data_log.csv"
log_file_processed_s2_data = log_dir / "processed_s2_data_log.csv"

# Result files
non_redundant_obs_file = results_dir / "non_redundant_observables.csv"
pruned_metrics_file = results_dir / "pruned_metrics.xlsx"
top3_metrics_file = results_dir / "top3_metrics.xlsx"
bottom3_metrics_file = results_dir / "bottom3_metrics.xlsx"
top3_metrics_per_cap_file = results_dir / "top3_metrics_per_capital.xlsx"
bottom3_metrics_per_cap_file = results_dir / "bottom3_metrics_per_capital.xlsx"
key_indicator_performance_file = results_dir / "key_indicator_performance_ranking.xlsx"

all_corrmat_file = results_dir / "corrmat_all.pdf"
pruned_corrmat_file = results_dir / "corrmat_pruned.pdf"
n_key_indicators_per_performance_plot = results_dir / "n_key_indicators_per_performance_group.png"

compact_metrics_filename = "monet2030_metrics.csv" 
compact_cis_filename = "monet2030_confintervals.csv"

# URLs
url_all_monet2030_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html'
    