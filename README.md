# Project: SIPI | Subproject: Analysis of beyond-GDP and beyond-profit metrics

## Introduction
The Swiss Impact & Prosperity Initiative (SIPI) is reimagining success for Switzerland. It goes beyond GDP and traditional business metrics, aligning economic vitality with societal well-being, ecological integrity, and long-term resilience. SIPI's vision is a Switzerland where prosperity is measured not only by financial growth, but by the resilience of communities, the health of ecosystems, and the inclusive opportunities that drive long-term value creation.

In this repository the code for analysis of existing beyond-GDP metrics is developed. The goal is to extract a small number of highly expressive, mutually minimally correlated metrics.

## User Guide/Technical Instructions
### How to Build the Docker Image
1. Clone the github repository to your system.
2. If you have not already done so, download and install Docker (https://www.docker.com/get-started/). If your working under MacOS or Windows, I recommend to use Docker Desktop.
3. Run docker (e.g. by launching docker desktop)
4. In your command line/terminal, check that docker runs properly by executing "docker version". You should see both Client and Server versions.
5. Cd into the project repository "sipi-data-analysis"
6. Build the docker image by running ```docker build -t sipi-data-analysis .```. You should now see how the system and python packages are being installed.

### How to run Jupyter Lab inside the Docker Image
The dockerfile exposes jupyter lab to the port 8888. Therefore, you can launch juypter lab by running

```docker run -p 8888:8888 -v $(pwd):/analysis_app sipi-data-analysis```

from inside the sipi-data-analysis project folder. In your command line/terminal you should now see jupyter running. In order to open jupyter lab in your browser just copy-paste the URL including the access token to a new browser tab.

REMARK: Make sure you include the ```-v $(pwd):/analysis_app``` flag as otherwise any modifcations made in the jupyter notebook won't be visible from outside the docker container. As a result, you won't be able to easily use Git to track your edits. Including this flag will mounts the current host directory (i.e. "$(pwd)") to the container path "/analysis_app", so all the changes inside the Docker container will be saved directly to your real local file system.

### How to run the Analysis Pipeline in the Main Jupyter Notebooks.
In order to simply run the pipeline as a user, follow these steps:

1. Start Docker + Jupyter (see instructions above)
2. Open the main jupyter notebooks (i.e. monet_data_analysis.ipynb or wise_data_analysis.ipynb) 
3. Make sure to set all the parameters governing (over)writing files as per your requirements. See help/docstrings for all the functions and classes if needed.
4. Run all cells.
5. Verify outputs.

### How to stop running your docker
1. In your command line, run `docker ps` to list all running docker containers. Copy the hash value in the NAME column of the container you want to stop.
2. Execute `docker stop [hash-value]` in your command line.

## Developer Guide
For information on how to augment, extend and build on the existing code base, please read chapter 5 of the PDF document "Documentation_SIPI_Data_Analysis_For_EPS.pdf".

## Contribution Guide


## References & Credits
We do not have ownership of the raw data used in this project. The WISE data is publically accessible [here](https://springernature.figshare.com/articles/dataset/A_comprehensive_Beyond-GDP_database_to_accelerate_wellbeing_inclusion_and_sustainability_research/26970415?file=49085821) and credits go to Liu, Wang, Behrens, et al. (see Liu, Kedi; Wang, Ranran; Behrens, Paul; Schrijver, Inge; Jansen, Annegeke; Rum, Irlan; et al. (2024). A comprehensive Beyond-GDP database to accelerate wellbeing, inclusion, and sustainability research. figshare. Dataset. [DOI](https://doi.org/10.6084/m9.figshare.26970415.v1)). The MONET2030 data from the Swiss Federal Statistical Office can be openly accessed [here](https://www.bfs.admin.ch/bfs/en/home/statistics/sustainable-development/monet-2030/all-indicators.html).
