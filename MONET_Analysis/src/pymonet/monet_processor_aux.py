class DataProcessor(object):
    def __init__(self):

   

    

    
    def compactify(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compactify all metrics from stage-2-processing
        into a single table.
        """
        # Split metrics from confidence intervals
        metrics = [d for d in self.processed_data_list["stage2"] if d["metric_id"].endswith("metr")]
        cis = [d for d in self.processed_data_list["stage2"] if d["metric_id"].endswith("ci")]
        
        metr_df_list = []
        ci_df_list = []
        for metric_dict in metrics:
            data = metric_dict["data"]
            data = self._standardize_colnames(data, metric_dict["metric_id"])
            data = self._integerize_year_ranges(data)
            metr_df_list.append(data.copy())

        for ci_dict in cis:
            data = ci_dict["data"]
            data = self._standardize_colnames(data, ci_dict["metric_id"])
            data = self._integerize_year_ranges(data)
            ci_df_list.append(data)
        
        compact_metric_df = metr_df_list[0]
        for df in metr_df_list[1:]:
            if df.columns[0] in compact_metric_df.columns:
                continue
            compact_metric_df = compact_metric_df.merge(df, how="outer", left_index=True, right_index=True)

        compact_ci_df = ci_df_list[0]
        for df in ci_df_list[1:]:
            if df.columns[0] in compact_ci_df.columns:
                continue
            compact_ci_df = compact_ci_df.merge(df, how="outer", left_index=True, right_index=True)

        return compact_metric_df, compact_ci_df


    


 

            

        # 4) STAGE-3-PROCESSED DATA
        # -------------------------
        compact_metrics, compact_cis = self.compactify()
        dirpath = self.processed_fpath / "stage_3"
        dirpath.mkdir(parents=True, exist_ok=True)

        # Make data available
        self.processed_data_list["stage3"]["metrics"] = compact_metrics
        self.processed_data_list["stage3"]["confidence_intervals"] = compact_cis

        # Write processed data to csv files
        compact_metrics.to_csv(dirpath / const.compact_metrics_filename)
        compact_cis.to_csv(dirpath / const.compact_cis_filename)


  




### READ PROCESSED DATA ###

          # Stage-1-processed data
        # ----------------------
        print("Reading stage-1-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.processed_fpath / "stage_1").glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.processed_data_list["stage1"].append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

        # Stage-2-processed data
        # ----------------------
        print("Reading stage-2-processed data from disk...")
        sorted_json_files = sorted([file for file in (self.processed_fpath / "stage_2").glob("*.json")])
        for file in sorted_json_files:
            with open(file, 'r') as f:
                loaded_dict = json.load(f)

            self.processed_data_list["stage2"].append({k: aux.deserialize_value(v) for k, v in loaded_dict.items()})
            
        print("-> done!")

        # Stage-3-processed data
        # ----------------------
        print("Reading stage-3-processed data from disk...")
        self.processed_data_list["stage3"]["metrics"] = pd.read_csv(self.processed_fpath / "stage_3" / const.compact_metrics_filename)
        self.processed_data_list["stage3"]["confidence_intervals"] = pd.read_csv(self.processed_fpath / "stage_3" / const.compact_cis_filename)

        self.processed_data_list["stage3"]["metrics"].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.processed_data_list["stage3"]["metrics"].set_index("Year", inplace=True)
        self.processed_data_list["stage3"]["confidence_intervals"].rename({"Unnamed: 0": "Year"}, axis=1, inplace=True)
        self.processed_data_list["stage3"]["confidence_intervals"].set_index("Year", inplace=True)
        print("-> done!")
