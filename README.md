# Project: SIPI | Subproject: Analysis of beyond-GDP and beyond-profit metrics

## Introduction


## How to Build the Docker Image

1. Clone the github repository to your system.
2. If you have not already done so, download and install Docker (https://www.docker.com/get-started/). If your working under MacOS or Windows, I recommend to use Docker Desktop.
3. Run docker (e.g. by launching docker desktop)
4. In your command line/terminal, check that docker runs properly by executing "docker version". You should see both Client and Server versions.
5. Cd into the project repository "sipi-data-analysis"
6. Build the docker image by running ```docker build -t sipi-data-analysis .```. You should now see how the system and python packages are being installed.

## How to run Jupyter Lab inside the Docker Image
The dockerfile exposes jupyter lab to the port 8888. Therefore, you can launch juypter lab by running

```docker run -p 8888:8888 sipi-data-analysis```

from inside the sipi-data-analysis project folder. In your command line/terminal you should now see jupyter running. In order to open jupyter lab in your browser just copy-paste the URL including the access token to a new browser tab.
