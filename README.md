# PandemicDataPrivacy
## Overview
Responding to a pandemic requires the timely dissemination of relevant health information. However, current patient de-identification methods do not account to the dynamics of infectious disease data, and their retrospective approach requires too much time to implement. To address these challenges, we developed a privacy risk estimation framework. The code contained in this repository uses the framework - which employs Monte Carlo random sampling methods - to forecast the privacy risk of different data sharing policies. The forecasted privacy risk estimates allow for data owners to dyanmically adjust the data sharing policy to share surveillance in near-real time while preserving patient privacy. The framework requires times series data of the disease cases of interest as well as the demographic joint statistics of the underlying population. These data can be derived from public datasets but we not provide such data herein. We do, however, provide sythetic example data, which we use in the demos to illustrate the frameworks technical details.
## Contents
 [source_code]: (source_code) code of framework implementation
 [data](data): Example datasets used in demos.
 [demos](demos): Multiple jupyter notebook demonstrations of the framework's code and implementation.
##
