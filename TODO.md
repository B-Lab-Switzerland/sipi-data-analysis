# TO DO LIST FOR SIPI DATA ANALYSIS PROJECT

This TODO lists tracks the progress as well as lists the
ideas for future work in the context of the sipi-data-analysis
project.

### Thoughts and Ideas
- The data processing pipeline is currently linear by design. In principle this could be generalized
  to a graph-structure (DAG) where the data transformation execution follows a selected path from the
  root node (raw data) to a selected leaf node (final result). As of today (July 29, 2025) it is not
  obvious if such a generalization is necessary or even helpful.


### Todo

- [ ] Add scraper to automatically get list of MONET2030 key indicators
- [ ] Add scraper for information about desired trends (should an indicator go up or down)?
- [ ] Add scraper for indicator goals (what is the goal for MONET2030 indicator XY?)
- [ ] Data analysis for "Cercle Indicateur" (CI) data
  - [ ] Add scraper for CI data (https://www.bfs.admin.ch/bfs/de/home/statistiken/nachhaltige-entwicklung/cercle-indicateurs.html)
  - [ ] Analyze CI data
- [ ] Improve data logging pipeline to keep save the data at every step along the data processing pipeline
- [ ] Add a proper logger 
- [ ] Augment/enhance/improve importance analysis of MONET metrics  
  - [ ] Compute summary statistics for each time series (mean, std, IQR, etc.)
  - [ ] Perform autocorrelation analysis
  - [ ] Check if time series can be clustered (cluster analysis, T-SNE, hierarchical clustering, etc.)
- [ ] Re-align the WISE analysis with the MONET analysis


### In Progress
- [ ] Augment/enhance/improve importance analysis of MONET metrics
- [ ] Complete documentation (in-code and external)

### Done ✓

- [x] Implement 1st version of WISE data analysis 
- [x] Implement scraper for MONET2030 indicators 
- [x] Adding per-capitals view to MONET analysis
- [x] Extract three best and worst performing indicators
- [x] Streamline data processing pipeline
- [x] Create table keeping track of which values were imputed and which ones weren't. 
- [x] Create a metric ID-to-name map
- [x] Augment/enhance/improve importance analysis of MONET metrics  
  - [x] Added standard data scaler
