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
key_indicators_fpath = meta_dir / "key_indicators.csv"
indicator_table_path = meta_dir / "indicator_table.csv"
metainfo_table_path = meta_dir / "monet_datafile_summary_table.csv"
metrics_meta_table_fpath = meta_dir / "metrics_meta_table.csv"


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
clean_vs_raw_plot_fpath = "clean_vs_raw.png"
                   
# (Stage 5 - Data imputation)
interp_data_fname = "monet2030_interpolated.csv"
envlp_data_fname = "monet2030_uncertainty_envelopes.csv"
interp_tracker_fname = "monet2030_interpolation_tracker.csv"
interpolated_vs_clean_plot_fpath = "interpolated_vs_clean.png"

# (Stage 6 - Time series decomposition)
p_values_fname = "stl_p_values.csv" 
optimal_stl_info_fname = "optimal_stl.csv"
trends_fname = "monet2030_trends.csv"
residuals_fname = "monet2030_residuals.csv"
trends_vs_interpolated_plot_fpath = "trend_vs_interpolated.png"
residuals_plot_fpath = "residuals.png"

# (Stage 7 - Scaling)
scaled_ts_fname = "monet2030_scaled_ts.csv"
scaled_resids_fname = "monet2030_scaled_residuals.csv"
zscores_plot_fpath = "zscores.png"

# Result files
# ------------
# (data availability analysis / data_availability_dir)
# - Number of metrics per capital
n_metrics_per_cap_fpath = data_availability_dir / "n_metrics_per_capital.csv"
n_metrics_per_cap_plot_fpath = data_availability_dir / "n_metrics_per_capital.png"

# - Number of sparse metrics per capital
sparse_metrics_analysis_fpath = data_availability_dir / "n_sparse_metrics_per_capital.csv"
n_sparse_by_capital_plot_fpath = data_availability_dir / "n_sparse_metrics_per_capital.png"

# - Number of irrelevant metrics per capital
irrelevant_metrics_analysis_fpath = data_availability_dir / "n_irrelevant_metrics_per_capital.csv"
n_irrelevant_by_capital_plot_fpath = data_availability_dir / "n_irrelevant_metrics_per_capital.png"

# - Number of datapoints per metric
n_measurements_per_metrics_fpath = data_availability_dir / "n_datapoints_per_metric.csv"
data_availability_map_fpath = data_availability_dir / "data_availability_map.csv"
data_availability_chart_fpath = data_availability_dir / "data_availability_all.pdf"

# (time series analysis / tsa_dir)
stationary_ts_fpath = tsa_dir / "stationary.csv"
non_stationary_ts_fpath = tsa_dir / "non_stationary.csv"

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

# - key indicator-associated metrics
key_indicator_ranking_fpath = perfa_dir / "key_indicators_performance_ranking.csv"
key_indicator_ranking_plot_fpath = perfa_dir / "key_indicators_performance_ranking_plot.png"
key_indicator_slope_stats_fpath = perfa_dir / "key_indicators_slope_norm_stats.csv"
key_indicator_slope_distro_plot_fpath = perfa_dir / "key_indicators_slope_norm_distribution.png"
key_indicator_top_performers_fpath = perfa_dir / "key_indicators_top_performers.xlsx"
key_indicator_bottom_performers_fpath = perfa_dir / "key_indicators_worst_performers.xlsx"
n_key_indicators_per_performance_plot_fpath = perfa_dir / "n_key_indicators_per_performance_group.png"


# ====
# URLs
# ====
url_all_monet2030_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html'
url_monet2030_key_indicators = 'https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/key-indicators.html'
