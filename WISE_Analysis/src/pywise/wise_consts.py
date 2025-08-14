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

# Raw data files
# --------------
wise_db_zippath = raw_dir / "WISE_Database.zip"
wise_db_fpath = raw_dir / "WISE_Database/Data/WISE_Database/WISE_Database.xlsx"
single_country_wise_db_fpath = lambda iso3: raw_dir / f"WISE_Database/Data/WISE_Database/WISE_Database_{iso3}.xlsx" 

# Meta files
# ----------
capmap_path = meta_dir / "capitals_map.csv"
wise_metatable_fname = meta_dir / "wise_metainfo_table.csv"

# Log files
# ---------
log_file_raw_data = log_dir / "raw_data_log.csv"
log_file_processed_s1_data = log_dir / "processed_s1_data_log.csv"
log_file_processed_s2_data = log_dir / "processed_s2_data_log.csv"

# Data files at different transformation stages
# ---------------------------------------------
# (Stage 1)
wise_metric_tables_fname = "wise_metrics.xlsx" 


# (Stage 4 - Data cleaning)
clean_data_fname = "wise_clean.xlsx"
duplicated_rows_fname = "duplicated_rows_removed.xlsx"
constant_cols_fname = "constant_metrics.xlsx"
outside_years_fname = "outside_year_range.xlsx"
sparse_cols_fname = "sparse_metrics.xlsx"
clean_vs_raw_plot_fpath = "clean_vs_raw.png"
                   
# (Stage 5 - Data imputation)
interp_data_fname = "wise_interpolated.xlsx"
envlp_data_fname = "wise_uncertainty_envelopes.xlsx"
interp_tracker_fname = "wise_interpolation_tracker.xlsx"
interpolated_vs_clean_plot_fpath = "interpolated_vs_clean.png"

# (Stage 6 - Time series decomposition)
p_values_fname = "stl_p_values.xlsx" 
optimal_stl_info_fname = "optimal_stl.xlsx"
trends_fname = "wise_trends.xlsx"
residuals_fname = "wise_residuals.xlsx"
trends_vs_interpolated_plot_fpath = "trend_vs_interpolated.png"
residuals_plot_fpath = "residuals.png"

# (Stage 7 - Scaling)
scaled_ts_fname = "wise_scaled_ts.xlsx"
scaled_resids_fname = "wise_scaled_residuals.xlsx"
zscores_plot_fpath = "zscores.png"

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
wise_download_url = "https://springernature.figshare.com/ndownloader/files/49085821/WISE_Database.zip"