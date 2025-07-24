# TO DO LIST FOR SIPI DATA ANALYSIS PROJECT

This TODO lists tracks the progress as well as lists the
ideas for future work in the context of the sipi-data-analysis
project.


### Todo

- [ ] Create table keeping track of which values were imputed and which ones weren't. Include information about imputation method as well as uncertainty (if available)  
- [ ] Add scraper to automatically get list of MONET2030 key indicators
- [ ] Add scraper for information about desired trends (should an indicator go up or down)?
- [ ] Add scraper for indicator goals (what is the goal for MONET2030 indicator XY?)
- [ ] Data analysis for "Cercle Indicateur" (CI) data
  - [ ] Add scraper for CI data (https://www.bfs.admin.ch/bfs/de/home/statistiken/nachhaltige-entwicklung/cercle-indicateurs.html)
  - [ ] Analyze CI data
- [ ] Improve data logging pipeline to keep save the data at every step along the data processing pipeline
- [ ] Add a proper logger 
- [ ] Augment/enhance/improve importance analysis of MONET metrics  
  - [ ] Add data scaler (in order to bring all data to same scale, e.g. [0,1])
  - [ ] Compute summary statistics for each time series (mean, std, IQR, etc.)
  - [ ] Perform autocorrelation analysis
  - [ ] Check if time series can be clustered (cluster analysis, T-SNE, hierarchical clustering, etc.)
- [ ] Re-align the WISE analysis with the MONET analysis
- [ ] Complete documentation (in-code and external)


### In Progress
- [ ] Streamline data processing pipeline

### Done ✓

- [x] Implement 1st version of WISE data analysis 
- [x] Implement scraper for MONET2030 indicators 
- [x] Adding per-capitals view to MONET analysis
- [x] Extract three best and worst performing indicators

