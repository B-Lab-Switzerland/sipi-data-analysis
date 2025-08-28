from pathlib import Path
import pytz

# =========
# Constants
# =========
# Project-wide timezone definition
# --------------------------------
zurich_tz = pytz.timezone("Europe/Zurich")

class WisePaths(object):
    def __init__(self, iso3):
        # ==========
        # File paths
        # ==========
        # Directories
        # -----------
        self.data_dir = Path("../data")
        self.meta_dir = self.data_dir / "metadata"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / f"processed/{iso3}"
        self.log_dir = self.data_dir / "logs"
        
        self.results_dir = Path(f"../results/{iso3}")
        self.tsa_dir = self.results_dir / "time_series_analysis"
        self.corra_dir = self.results_dir / "correlation_analysis"
        self.perfa_dir = self.results_dir / "performance_analysis"
        self.data_availability_dir = self.results_dir / "data_availability_analysis"
        
        # Raw data files
        # --------------
        self.wise_db_zippath = self.raw_dir / "WISE_Database.zip"
        self.wise_db_fpath = self.raw_dir / "WISE_Database/Data/WISE_Database/WISE_Database.xlsx"
        self.single_country_wise_db_fpath = lambda iso3: self.raw_dir / f"WISE_Database/Data/WISE_Database/WISE_Database_{iso3}.xlsx" 
        
        # Meta files
        # ----------
        self.capmap_path = self.meta_dir / "capitals_map.csv"
        self.wise_metatable_fname = self.meta_dir / "wise_metainfo_table.csv"
        self.trend_directions = self.meta_dir / "trend_directions.csv"
        
        # Log files
        # ---------
        self.log_file_raw_data = self.log_dir / "raw_data_log.csv"
        self.log_file_processed_s1_data = self.log_dir / "processed_s1_data_log.csv"
        self.log_file_processed_s2_data = self.log_dir / "processed_s2_data_log.csv"
        
        # Data files at different transformation stages
        # ---------------------------------------------
        # (Stage 1)
        self.wise_metric_tables_fname = "wise_metrics.xlsx" 
        
        # (Stage 2 - Data cleaning)
        self.clean_data_fname = "wise_clean.xlsx"
        self.duplicated_rows_fname = "duplicated_rows_removed.xlsx"
        self.constant_cols_fname = "constant_metrics.xlsx"
        self.outside_years_fname = "outside_year_range.xlsx"
        self.sparse_cols_fname = "sparse_metrics.xlsx"
        self.clean_vs_raw_plot_fpath = "clean_vs_raw.png"
                           
        # (Stage 3 - Data imputation)
        self.interp_data_fname = "wise_interpolated.xlsx"
        self.envlp_data_fname = "wise_uncertainty_envelopes.xlsx"
        self.interp_tracker_fname = "wise_interpolation_tracker.xlsx"
        self.interpolated_vs_clean_plot_fpath = "interpolated_vs_clean.png"
        
        # (Stage 4 - Time series decomposition)
        self.p_values_fname = "stl_p_values.xlsx" 
        self.optimal_stl_info_fname = "optimal_stl.xlsx"
        self.trends_fname = "wise_trends.xlsx"
        self.residuals_fname = "wise_residuals.xlsx"
        self.trends_vs_interpolated_plot_fpath = "trend_vs_interpolated.png"
        self.residuals_plot_fpath = "residuals.png"
        
        # (Stage 5 - Scaling)
        self.scaled_ts_fname = "wise_scaled_ts.xlsx"
        self.scaled_resids_fname = "wise_scaled_residuals.xlsx"
        self.zscores_plot_fpath = "zscores.png"
        
        # Result files
        # ------------
        # (data availability analysis / data_availability_dir)
        # - Number of metrics per capital
        self.n_metrics_per_cap_fpath = self.data_availability_dir / "n_metrics_per_capital.xlsx"
        self.n_metrics_per_cap_plot_fpath = self.data_availability_dir / "n_metrics_per_capital.png"
        
        # - Number of sparse metrics per capital
        self.sparse_metrics_analysis_fpath = self.data_availability_dir / "n_sparse_metrics_per_capital.xlsx"
        self.n_sparse_by_capital_plot_fpath = self.data_availability_dir / "n_sparse_metrics_per_capital.png"
        
        # - Number of irrelevant metrics per capital
        self.irrelevant_metrics_analysis_fpath = self.data_availability_dir / "n_irrelevant_metrics_per_capital.xlsx"
        self.n_irrelevant_by_capital_plot_fpath = self.data_availability_dir / "n_irrelevant_metrics_per_capital.png"
        
        # - Number of datapoints per metric
        self.n_measurements_per_metrics_fpath = self.data_availability_dir / "n_datapoints_per_metric.xlsx"
        self.data_availability_map_fpath = self.data_availability_dir / "data_availability_map.xlsx"
        self.data_availability_chart_fpath = self.data_availability_dir / "data_availability_all.pdf"
        
        
        # (time series analysis / tsa_dir)
        self.stationary_ts_fpath = self.tsa_dir / "stationary.xlsx"
        self.non_stationary_ts_fpath = self.tsa_dir / "non_stationary.xlsx"
        
        # (correlation analysis / corra_dir)
        self.all_corrmat_fpath = lambda infix: self.corra_dir / f"corrmat_all_{infix}.csv"
        self.all_corrmat_plot_fpath = lambda infix: self.corra_dir / f"corrmat_all_{infix}.pdf"
        self.metric_counts_fpath = lambda infix:self.corra_dir / f"n_to_keep_vs_corr_threshold_{infix}.csv"
        self.metric_counts_plot_fpath = lambda infix: self.corra_dir / f"n_nonredundant_per_threshold_{infix}.pdf"
        self.to_keep_fpath = lambda infix: self.corra_dir / f"metrics_to_keep_{infix}.xlsx"
        self.corr_groups_fpath = lambda infix, thstring: self.corra_dir / f"correlation_groups_{infix} / corr_group_th{thstring}.xlsx"
        self.corr_val_distribution_plot_fpath = lambda infix: self.corra_dir / f"corr_val_distribution_{infix}.png"
        self.n_retained_key_metrics_fpath = lambda infix: self.corra_dir / f"n_retained_key_metrics_after_pruning_{infix}.csv"
        self.n_retained_key_metrics_plot_fpath = lambda infix: self.corra_dir / f"n_retained_key_metrics_after_pruning_{infix}.png"
        
        # (performance analysis / perfa_dir)
        # - all metrics
        self.ranking_fpath = self.perfa_dir / "performance_ranking.csv"
        self.ranking_plot_fpath = self.perfa_dir / "performance_ranking_plot.png"
        self.slope_stats_fpath = self.perfa_dir / "slope_norm_stats.csv"
        self.slope_distro_plot_fpath = self.perfa_dir / "slope_norm_distribution.png"
        self.top_performers_fpath = self.perfa_dir / "top_performers.xlsx"
        self.bottom_performers_fpath = self.perfa_dir /  "worst_performers.xlsx"
        
        # ====
        # URLs
        # ====
        self.wise_download_url = "https://springernature.figshare.com/ndownloader/files/49085821/WISE_Database.zip"

# module-global "active" WisePaths
_active_paths: WisePaths | None = None

def configure_paths(**kwargs):
    """
    Set up the global WisePaths instance.
    """
    global _active_paths
    _active_paths = WisePaths(**kwargs)

def __getattr__(name: str):
    """
    Dynamic access to attributes of the active WisePaths.
    """
    if _active_paths is None:
        raise AttributeError(
            f"Paths are not configured yet. Call pyconsts.configure_paths(...) first."
        )
    if hasattr(_active_paths, name):
        return getattr(_active_paths, name)
    raise AttributeError(name)

def __dir__():
    base = list(globals())
    if _active_paths is not None:
        base.extend(vars(_active_paths).keys())
    return sorted(set(base))