# PandemicDataPrivacy
## Overview
Responding to a pandemic requires the timely dissemination of relevant health information. However, current patient de-identification methods do not account to the dynamics of infectious disease data, and their retrospective approach requires too much time to implement. To address these challenges, we developed a privacy risk estimation framework. The code contained in this repository uses the framework - which employs Monte Carlo random sampling methods - to forecast the privacy risk of different data sharing policies. The forecasted privacy risk estimates allow for data owners to dyanmically adjust the data sharing policy to share surveillance in near-real time while preserving patient privacy. The framework requires times series data of the disease cases of interest as well as the demographic joint statistics of the underlying population. These data can be derived from public datasets but we not provide such data herein. We do, however, provide sythetic example data, which we use in the [demos](demos) to illustrate the framework's technical details.
## Repo Contents
 [source_code](source_code): code of framework implementation
 [data](data): Example datasets used in demos.
 [demos](demos): Multiple jupyter notebook demonstrations of the framework's code and implementation.
## System Requirements
The code was developed and tested in Python version 3.8. Python package requirements: itertools, numpy, and pandas. The software has been tested on Mac OSX 11.x.
## Installation Guide
The software can be run after downloading the repo files. The typical install time is relatively short (<5 minutes).
## Demo
The demos (.ipynb files) can be run in jupyter notebook. To avoid the need to change the notebook contents, the two .csv files in the data folder should be in the same directory as the .ipynb files.
Each demonstration file includes expected outputs for each cell and expected run times for the most computationally cells. The run times were calculated on 2019 MacBook Pro with a 2.4 GHz quad-core processor and 16GB of RAM.
## Instructions for use
The demo notebooks run the software on the example data. The same notebooks could be used to reproduce the results in the manuscript by replacing the example data with the Johns Hopkins University COVID-19 surveillance data and the 2010 US Census PCT12A-I tables.
