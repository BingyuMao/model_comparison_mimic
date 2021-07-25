# Model comparison for in hospital mortality prediction and survival analysis based on MIMIC III

## Introduction

This is the code repository for BMI 6319 course project in Spring 2021.

### The overarching goal of this research:

To compare the results of different models on different inputs, then reach out that if add demographics information will improve the results of in hospital mortality/survival analysis for different types of models. After completing this project, we will know which model works better on EHR data especially on MIMIC III for predicting in hospital mortality/ survival analysis and if demographics information has an obvious effect on the accuracy of prediction. In this way, it will be helpful for future studies on model/input selection.


Subgoals are:
- Descriptive analysis on MIMIC III dataset with plots;

- Use MIMIC III dataset to compare different models such as the statistics model Logistic Regression with machine learning models Light Gradient Boosting Machine and Recurrent Neural Network on in hospital mortality prediction based on (1) diagnosis, prescription and procedure, (2) diagnosis, prescription, procedure and demographics.

- Use MIMIC III dataset to compare different machine learning models such as Random Survival Forest and Recurrent Neural Network for survival analysis based on (1) diagnosis, prescription and procedure, (2) diagnosis, prescription, procedure and demographics.


### The dataset used:

MIMIC III with ADMISSIONS, DIAGNOSES_ICD, ICUSTAYS, PATIENTS, PRESCRIPTIONS and PROCEDURES tables.

I'll not provide the MIMIC III data itself, you need to acquire the data yourself from https://mimic.physionet.org/.


## Steps to run this project

You may choose to start from the Model part directly with data files after data pre-process, or you can also start from Data pre-process with the raw data.

### Data pre-process

1. The [DescriptiveAnalysis.ipynb](https://github.com/BingyuMao/model_comparison_mimic/blob/main/DescriptiveAnalysis.ipynb) is the draft of descriptive analysis from the raw MIMIC III dataset.
2. The [DataPreprocess.py](https://github.com/BingyuMao/model_comparison_mimic/blob/main/DataPreprocess.py) is the first part of data pre-process, it will output two types of csv file for in hospital mortality prediction and survival analysis: case&control files contain diagnosis, procedures and prescriptions; case&control files contain diagnosis, prescriptions, procedures and demographics. For each type we will have two files: one is case contains died patients' information and the other is contorl contains other patients' information.

3. The [preprocessing.py](https://github.com/BingyuMao/model_comparison_mimic/blob/main/preprocessing.py) is the second part of data pre-process, it will output three types of files (train, valid and test) and in these files, we have a list of lists. Every list represents a patient and in this list, we will have his/her ID, other lists with different visits and for every time stamp we have diagnosis, prescription, etc.

### Model

1. For in hospital mortality task, you can run [Mortality_dp.ipynb](https://github.com/BingyuMao/model_comparison_mimic/blob/main/Mortality_dp.ipynb) which contains the three models for in hospital mortality prediction with diagnoses, procedures & prescriptions information; [Mortality_dpd.ipynb](https://github.com/BingyuMao/model_comparison_mimic/blob/main/Mortality_dpd.ipynb) which contains the three models with diagnoses, prescriptions, procedures & demographics information. The data used in these scripts can be found in the data floder.

2. For in survival analysis task, you can run [Survival_dp.ipynb](https://github.com/BingyuMao/model_comparison_mimic/blob/main/Survival_dp.ipynb) which contains the two models for survival analysis with diagnoses, procedures & prescriptions information; [Survival_dpd.ipynb](https://github.com/BingyuMao/model_comparison_mimic/blob/main/Survival_dpd.ipynb) which contains the two models with diagnoses, prescriptions, procedures & demographics information. The data used in these scripts can be found in the data floder.

3. The script [model.py](https://github.com/BingyuMao/model_comparison_mimic/blob/main/model.py) contains all the required functions for RNN model.


