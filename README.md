# ML_Heart_Disease_Project

## Contents
[Prerequisites](#prerequisites)
[devicespec.m](#devicespec.m)
[loadheart.m](#loadheart.m)
[RF_2_2018_10_04.m](#RF_2_2018_10_04.m)
[RF_PredImp_3_2018_10_10.m](#RF_PredImp_3_2018_10_10.m)
[Authors](#Authors)
[License](#License)



## Prerequisites
* Require Matlab version 2018b
* Installation of Parallel Computing Toolbox



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
Runs function scripts **devicespec.m** and **loadheart.m** to define runtime environment and to load data from heart.csv.
Contains the following sections:

1. Section 1
  Sets up 10-fold CV and performs RF analysis using Treebagger model. Calculates Misclassification Rate (mcr). Also generates confusion matrix based on the summation of the 10 test set predictions generated for each CV model.
2. Section 2
  Performs Bayesian optimisation on a treebagger model using all the data (Not using CV). Matlab bayesopt uses **Optimsation.m** function script to calculate mcr and to update the Acquisition function defining the argmax for the Optimisation Function. Current hyperparamters being explore are:
  * MinLeafSize (Minimum number of observations per tree leaf)
  * NumPredictorsToSample (Number of variables to select at random for each decision split)
3. Section 3
  Performs Bayesian optimisation USING k-fold cross validation. Calculates mcr over each fold sample and then calculates an average mcr over all k fold samples. Matlab bayesopt uses **myCVlossfcn.m** function script as its optimisation function to calculate mcr and to update the Acquisition function defining the argmax for the Optimisation Function.
Current hyperparamters being explore are:
  * MinLeafSize (Minimum number of observations per tree leaf)
  * NumPredictorsToSample (Number of variables to select at random for each decision split)
4. Section 4
 Performs hyperparameter grid search looking at hyperparameters MinLeafSize and NumPredictorsToSample. Uses **myCVlossfcn_grid.m** function script to generate mcr values.
 
## RF_PredImp_3_2018_10_10.m
Performs k-fold CV using RF analysis to calculate predictor importance for all included variables. Predictor importance is calculated for each k-fold sample and then average values are calulated for each predictor. Average values are then displayed in a bar chart. 


## Authors
Kevin Ryan

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
  




