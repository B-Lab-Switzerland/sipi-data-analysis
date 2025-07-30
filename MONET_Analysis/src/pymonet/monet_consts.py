from pathlib import Path
import pytz

# =========
# Constants
# =========
# Project-wide timezone definition
# --------------------------------
zurich_tz = pytz.timezone("Europe/Zurich")

# ==========
# File paths
# ==========
# Directories
# -----------
data_dir = Path("../data")
meta_dir = data_dir / "metadata"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
log_dir = data_dir / "logs"

results_dir = Path("../results")
tsa_dir = results_dir / "time_series_analysis"
corra_dir = results_dir / "correlation_analysis"
perfa_dir = results_dir / "performance_analysis"
data_availability_dir = results_dir / "data_availability_analysis"

# Meta files
# ----------
capmap_path = meta_dir / "capitals_map.csv"
trend_directions = meta_dir / "trend_directions.csv"
key_indicators = meta_dir / "key_indicators.csv"
indicator_table_path = meta_dir / "indicator_table.csv"
metainfo_table_path = meta_dir / "monet_datafile_summary_table.csv"

# Log files
# ---------
log_file_raw_data = log_dir / "raw_data_log.csv"
log_file_processed_s1_data = log_dir / "processed_s1_data_log.csv"
log_file_processed_s2_data = log_dir / "processed_s2_data_log.csv"

# Data files at different transformation stages
# ---------------------------------------------
# (Stage 1)
# multiple JSON files stored in processed_dir / "stage_1"

# (Stage 2)
# multiple JSON files stored in processed_dir / "stage_2"
metric_id2name_fname = "metric_id_to_name_map.csv"

# (Stage 3)
compact_metrics_filename = "monet2030_metrics.csv" 
compact_cis_filename = "monet2030_confintervals.csv"

# (Stage 4 - Data cleaning)
clean_data_fname = "monet2030_clean.csv"
irrelevant_metrics_fname = "agenda2030_irrelevant.csv"
duplicated_rows_fname = "duplicated_rows_removed.csv"
constant_cols_fname = "constant_metrics.csv"
outside_years_fname = "outside_year_range.csv"
sparse_cols_fname = "sparse_metrics.csv"
                   
# (Stage 5 - Data imputation)
interp_data_fname = "monet2030_interpolated.csv"
envlp_data_fname = "monet2030_uncertainty_envelopes.csv"
interp_tracker_fname = "monet2030_interpolation_tracker.csv"

# (Stage 6 - Time series decomposition)
p_values_fname = "stl_p_values.csv" 
optimal_stl_info_fname = "optimal_stl.csv"
trends_fname = "monet2030_trends.csv"
residuals_fname = "monet2030_residuals.csv"

# (Stage 7 - Scaling)
scaled_ts_fname = "monet2030_scaled_ts.csv"
scaled_resids_fname = "monet2030_scaled_residuals.csv"

# Result files
# ------------
# (data availability analysis / data_availability_dir)
n_sparse_by_capital_plot_fpath = data_availability_dir / "n_sparse_metrics_by_capital.png"
n_irrelevant_by_capital_plot_fpath = data_availability_dir / "n_irrelevant_metrics_by_capital.png"
data_availability_barchart_fpath = data_availability_dir / "data_availability_barchart.png"
sparse_metrics_analysis_fpath = data_availability_dir / "sparse_metrics.csv"
irrelevant_metrics_analysis_fpath = data_availability_dir / "irrelevant_metrics.csv"

# (time series analysis / tsa_dir)
stationary_ts_fpath = tsa_dir / "stationary.csv"
non_stationary_ts_fpath = tsa_dir / "non_stationary.csv"

# (correlation analysis / corra_dir)
non_redundant_obs_fpath = corra_dir / "non_redundant_observables.csv"
pruned_metrics_fpath = corra_dir / "pruned_metrics.xlsx"
all_corrmat_fpath = corra_dir / "corrmat_all.pdf"
pruned_corrmat_fpath = corra_dir / "corrmat_pruned.pdf"

# (performance analysis / perfa_dir)
top3_metrics_fpath = perfa_dir / "top3_metrics.xlsx"
bottom3_metrics_fpath = perfa_dir / "bottom3_metrics.xlsx"
top3_metrics_per_cap_fpath = perfa_dir / "top3_metrics_per_capital.xlsx"
bottom3_metrics_per_cap_fpath = perfa_dir / "bottom3_metrics_per_capital.xlsx"
key_indicator_performance_fpath = perfa_dir / "key_indicator_performance_ranking.xlsx"
n_key_indicators_per_performance_plot_fpath = perfa_dir / "n_key_indicators_per_performance_group.png"


# ====
# URLs
# ====
url_all_monet2030_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html'
