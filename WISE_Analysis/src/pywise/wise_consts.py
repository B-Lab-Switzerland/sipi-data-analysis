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
# - Number of metrics per capital
n_metrics_per_cap_fpath = data_availability_dir / "n_metrics_per_capital.xlsx"
n_metrics_per_cap_plot_fpath = data_availability_dir / "n_metrics_per_capital.png"

# - Number of sparse metrics per capital
sparse_metrics_analysis_fpath = data_availability_dir / "sparse_metrics.xlsx"
n_sparse_by_capital_plot_fpath = data_availability_dir / "n_sparse_metrics_by_capital.png"

# - Number of irrelevant metrics per capital
irrelevant_metrics_analysis_fpath = data_availability_dir / "irrelevant_metrics.xlsx"
n_irrelevant_by_capital_plot_fpath = data_availability_dir / "n_irrelevant_metrics_by_capital.png"

# - Number of datapoints per metric
n_measurements_per_metrics_fpath = data_availability_dir / "n_datapoints_per_metric.xlsx"
data_availability_map_fpath = data_availability_dir / "data_availability_map.xlsx"
data_availability_chart_fpath = data_availability_dir / "data_availability_all.pdf"


# (time series analysis / tsa_dir)
stationary_ts_fpath = tsa_dir / "stationary.xlsx"
non_stationary_ts_fpath = tsa_dir / "non_stationary.xlsx"

# (correlation analysis / corra_dir)
all_corrmat_fpath = lambda infix: corra_dir / f"corrmat_all_{infix}.csv"
all_corrmat_plot_fpath = lambda infix: corra_dir / f"corrmat_all_{infix}.pdf"
metric_counts_fpath = lambda infix:corra_dir / f"n_to_keep_vs_corr_threshold_{infix}.csv"
metric_counts_plot_fpath = lambda infix: corra_dir / f"n_nonredundant_per_threshold_{infix}.pdf"
to_keep_fpath = lambda infix: corra_dir / f"metrics_to_keep_{infix}.xlsx"
corr_groups_fpath = lambda infix, thstring: corra_dir / f"correlation_groups_{infix} / corr_group_th{thstring}.xlsx"
corr_val_distribution_plot_fpath = lambda infix: corra_dir / f"corr_val_distribution_{infix}.png"
n_retained_key_metrics_fpath = lambda infix: corra_dir / f"n_retained_key_metrics_after_pruning_{infix}.csv"
n_retained_key_metrics_plot_fpath = lambda infix: corra_dir / f"n_retained_key_metrics_after_pruning_{infix}.png"

# (performance analysis / perfa_dir)
# - all metrics
ranking_fpath = perfa_dir / "performance_ranking.csv"
ranking_plot_fpath = perfa_dir / "performance_ranking_plot.png"
slope_stats_fpath = perfa_dir / "slope_norm_stats.csv"
slope_distro_plot_fpath = perfa_dir / "slope_norm_distribution.png"
top_performers_fpath = perfa_dir / "top_performers.xlsx"
bottom_performers_fpath = perfa_dir /  "worst_performers.xlsx"

# ====
# URLs
# ====
wise_download_url = "https://springernature.figshare.com/ndownloader/files/49085821/WISE_Database.zip"