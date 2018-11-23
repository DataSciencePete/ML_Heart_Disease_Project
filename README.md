# ML_Heart_Disease_Project

## Contents
1. [Prerequisites](#prerequisites)
2. [Data folder](#data-folder)
3. [Exploratory Data Analysis (EDA folder)](#exploratory-data-analysis-eda-folder)
4. [NB tuning folder](#nb-tuning-folder)
5. [RF tuning folder](#rf-tuning-folder)
6. [Model Comparison folder](#model-comparison-folder)
7. [fitcnb_ce folder (adapted Matlab implementation of fitcnb using cross entropy)](#fitcnb_ce-folder-adapted-matlab-implementation-of-fitcnb-using-cross-entropy)
8. [utils folder](#utils-folder)
9. [Authors](#authors)
10. [License](#license)



## Prerequisites
* Matlab version 2018a
* Statistics toolbox

The following can be used but are skipped if not present:
* Parallel Computing Toolbox (required to optimise Random Forest computations for device)
* Deep learning toolbox (required to plot confusion charts)

## Data folder
* load_heart_csv.m

  Loads the datafile containing the heart data named `heart.csv` from the current directory and splits the data into training and test sets, returning the labels and features for the training and test sets as well as a `cvpartition` object. The script fixes a random seed so that the cross validation partition and split of test and training data are deterministic to allow for repeatability.
  
## Exploratory Data Analysis (EDA folder)

## NB tuning folder

* Run_NB_Analysis.m

  This is the top-level script running experiments on the Naive Bayes model. The script runs Bayesian optimisation and a grid search testing normal and kernel distributions and optimising kernel width on all features. A manual grid search is also run where all combinations of distribution were tried on continuous features. The data is standardised and the same process is re-run. The `Naive_Bayes_Optimisation.m` and `Naive_Bayes_man_gs.m` functions are called to run the optimisations.

* Naive_Bayes_Optimisation.m

Function to run Bayesian Optimisation and grid search given some features, labels and a cross validation partition. Prints the runtime for these optimisations to the terminal.

* Naive_Bayes_man_gs.m

Runs a manual grid search of all possible combinations of distribution over the set of features. Categorical features are fixed to the multivariable multinomial distribution `mvmm` and while kernel `kernel` and gaussian `normal` distributions are used for the continuous features.


## RF tuning folder

* feature_selection.m

  Performs k-fold CV using RF analysis to calculate predictor importance for all included variables. Predictor importance is calculated for each k-fold   sample and then average values are calulated for each predictor. Average values are then displayed in a bar chart. 
  
* loss_function_comparison.m

  Sets up 10-fold CV and performs RF analysis using the Matlab implementation Treebagger.
  Performs search for optimal Treebagger hyperparameters MinLeafSize (Minimum number of observations per tree leaf), NumPredictorsToSample (Number of variables to select at random for each decision split) and numTrees (number of trees to include in ensembl random forest) using Bayesian Optimisation.
  Performs 20 cycles of Bayesian Optimisation sampling 30 different points of the hyperparameter search space. 
  Error assessed using either MCR (**`lossFcn_RF_MCR.m`** found in RF tuning folder) or Cross Entropy (CE) (**`lossFcn_RF_CE.m`** found in RF tuning folder) loss functions. 
  Calculates mean and sd values for different performance metrics (recall, precision, F1, specificity, accuracy, AUC) over each 20 cycle run.
  Returns bar chart of mean performance metrics (recall, precision, F1, specificity, accuracy, AUC) and ROC curve showing best performing models    generated using MCR and CE.

## Model Comparison folder
* Model_Comparison.m 
  
  Trains optimised NB and RF models using optimised hyperparamters (optimal parameters are hard-coded for ease) generated from running Run_NB_Analysis.m for NB and feature_selection.m for RF.
  on the complete training dataset. Generated models are used to make predictions for the test set data. Performance metrics are generated (recall, precision, F1, specificity, accuracy, AUC) for each model and a bar chart is returned comparing the performance of NB and RF models on the test set data. A ROC curveis also generated comparing NB and RF models. 


## fitcnb_ce folder (adapted Matlab implementation of fitcnb using cross entropy)
Adapted Matlab's 'bayesopt' fitcnb implementation to use cross entropy instead of misclassification rate as the loss function used to explore the hyperparamter search space. The current version is compatible with R2018a and R2018B Matlab versions. The code used to change the loss function is located in createObjFcn_ce.m. All other paths to other scripts called within fitcnb implementation have been preserved so it retains all other fitcnb functionality.


## utils folder
* devicespec.m
  
  Function script which calculates specification for the machine being used to run the analysis and implements parallel pool environment.
  * Finds presence/absence of GPU.
  * Finds Number of cores available.
  * Finds number of CPUSs avaialable.
  * Implement a paralell pool environment using the available resources.
  
* get_performance.m
  
  Function script which takes as input a classification model, a confusion matrix for the inputted classification model, matrix of input features and a corresponding array of labels for the inputted features. Returns the following performance metrics 
  * recall
  * precision
  * F1
  * specificity
  * accuracy
  * AUC
  and also returns a ROC curve for the model

  
## Authors
Kevin Ryan, Peter Grimshaw

## License
This project is licensed under the MIT License - see the LICENSE.md file for details












  




