# ML_Heart_Disease_Project
## devicespec.m
Calculates specification for the machine being used to run the anlysis and implements paralell pool environment.

* Finds presence/absence of GPU.
* Finds Number of cores available.
* Finds number of CPUSs avaialable.
* Implement a paralell pool environment using the available resources.

## loadheart.m
Loads csv file on current machine located at /Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv
Returns:
* table where each column corresponds to a predictor variable - defined as variable 'In'
* array of target values - defined as variable 'Out'

## RF_2_2018_10_04.m
Runs function scripts devicespec.m and loadheart.m to define runtime environment and to load data from heart.csv.
Contains the following sections:

1. Section 1
  Sets up 10-fold CV and performs RF analysis using Treebagger model. Calculates Misclassification Rate (mcr). Also generates confusion matrix based on the summation of the 10 test set predictions generated for each CV model.
2. Section 2
  Performs Bayesian optimisation on a treebagger model using all the data (Not using CV). Uses Optimsation.m function script to calculate OOB error and to update the Acquisition function defining the argmax for the Optimisation Function.
## myCVlossfcn.m
## Optimisation.m


RF_PredImp_3_2018_10_10.m
