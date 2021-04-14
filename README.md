# Model comparison for in hospital mortality prediction and survival analysis based on MIMIC III

## Introduction


### The overarching goal of this research:

To compare the results of different models on different inputs, then reach out that if add demographics information will improve the results of in hospital mortality/survival analysis for different types of models. After completing this project, we will know which model works better on EHR data especially on MIMIC III for predicting in hospital mortality/ survival analysis and if demographics information has an obvious effect on the accuracy of prediction. In this way, it will be helpful for future studies on model/input selection.


Subgoals are:
- Descriptive analysis on MIMIC III dataset with plots;

- Use MIMIC III dataset to compare different models such as the statistics model Logistic Regression with machine learning models Light Gradient Boosting Machine and Recurrent Neural Network on in hospital mortality prediction based on (1) diagnosis, prescription and procedure, (2) diagnosis, prescription, procedure and demographics.

- Use MIMIC III dataset to compare different machine learning models such as Random Survival Forest and Recurrent Neural Network for survival analysis based on (1) diagnosis, prescription and procedure, (2) diagnosis, prescription, procedure and demographics.


### The dataset used:

MIMIC III with ADMISSIONS, DIAGNOSES_ICD, ICUSTAYS, PATIENTS, PRESCRIPTIONS and PROCEDURES tables.

I'll not provide the MIMIC III data itself, you need to acquire the data yourself from https://mimic.physionet.org/.


### The final deliverables:

- Plots for MIMIC III dataset descriptive analysis.

- AUROC scores and plots for different models on in hospital mortality prediction based on (1) diagnosis and prescriptions, (2) diagnosis, prescriptions and demographics.

- Concordance index and survival analysis plots for different models on survival analysis based on (1) diagnosis and prescriptions, (2) diagnosis, prescriptions and demographics.


### The metrics used to evaluate the data analysis:

AUROC (for in hospital mortality) and Concordance index (for survival analysis).


## Steps to run this project

You may choose to start from the Model part directly with data files after data pre-process, or you can also start from Data pre-process with the raw data.

