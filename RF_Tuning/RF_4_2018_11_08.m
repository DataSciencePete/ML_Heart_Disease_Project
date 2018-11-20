%% Load data into Matlab
% [train_features, train_labels, test_features, test_labels, X_header, cvp] = loadheart('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv'); % see script file loadhear.m
[train_features, train_labels, test_features, test_labels, X_header,cvp] = load_heart_csv('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv', 'table', 'categorical'); % see script file loadhear.m
par = devicespec(); % see script file devicespec.m


%% Section 2
% Prepare hyperparamter object for optimisation using Bayesian Optimisation

% Min No of Observations per leaf
maxMinLS = 50;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');


% Remove predictor variables from table In which have a mean Predictor
% Importance <0.4
In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

% No of variables to consider at each split in the tree
numPTS = optimizableVariable('numPTS',[1,size(In_high_imp_variables,2)],'Type','integer'); % define number of predictors

% Total number of trees generated to make random forest
maxNumTrees = 500;
numTrees = optimizableVariable('numTrees',[1,maxNumTrees],'Type','integer'); % define max number of trees to define forest

%Hyperparamter object
hyperparametersRF = [minLS; numPTS; numTrees];

%% Run Bayesian Optimisation using mcr as error value to evaluate objective 20 times and calculate mean and variance values for each hyperparameter

% Run bayesopt using mcr as error value to evaluate objective function 20 times and calculate the mean hyperparamter values for RF
% Initialise summary table
summary_table_MCR = [];
for i = 1:5
    % Rerun bayesian optimisation 20 times to generate a set of values from
    % which an average hyperparameter setting can be gleaned 
    results_paralllel_MCR = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 5, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    results_minLS_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.minLS;
    results_numPTS_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.numPTS;
    results_numTrees_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.numTrees;
    results_min_estmcr_MCR = results_paralllel_MCR.MinEstimatedObjective;
    results_total_train_time_MCR = results_paralllel_MCR.TotalElapsedTime;
    summary_table_MCR = [summary_table_MCR; results_minLS_MCR results_numPTS_MCR results_numTrees_MCR results_min_estmcr_MCR results_total_train_time_MCR];


end

mean_minLS_MCR = mean(summary_table_MCR(:,1));
variance_minLS_MCR = var(summary_table_MCR(:,1));
mean_numPTS_MCR = mean(summary_table_MCR(:,2));
variance_numPTS_MCR = var(summary_table_MCR(:,2));
mean_numTrees_MCR = mean(summary_table_MCR(:,3));
variance_numTrees_MCR = var(summary_table_MCR(:,3));
mean_min_estmcr_MCR = mean(summary_table_MCR(:,4));
variance_min_estmcr_MCR = var(summary_table_MCR(:,4));
mean_total_train_time_MCR = mean(summary_table_MCR(:,5));
variance_total_train_time_MCR = var(summary_table_MCR(:,5));


% Time how long it takes to perform 30 Random Forest iterations with Bayesian Optimisation. 
% This will be used as a comparison against the time taken to perform  30
% NB iterations with Bayesian Optimisation
tic;

results_paralllel_er0_5_MCR = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);

RF_hyperparameter_search_time = toc;

%% Run Bayesian Optimisation using ce as error value to evaluate objective 20 times and calculate mean and variance values for each hyperparameter

% Run bayesopt using ce as error value to evaluate objective function 20 times and calculate the mean hyperparamter values for RF
% Initialise summary table
summary_table_CE = [];
for i = 1:5
    % Rerun bayesian optimisation 20 times to generate a set of values from
    % which an average hyperparameter setting can be gleaned 
    results_paralllel_CE = bayesopt(@(params)myCVlossfcn_ce(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 5, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    results_minLS_CE = results_paralllel_CE.XAtMinEstimatedObjective.minLS;
    results_numPTS_CE = results_paralllel_CE.XAtMinEstimatedObjective.numPTS;
    results_numTrees_CE = results_paralllel_CE.XAtMinEstimatedObjective.numTrees;
    results_min_estce_CE = results_paralllel_CE.MinEstimatedObjective;
    results_total_train_time_CE = results_paralllel_CE.TotalElapsedTime;
    summary_table_CE = [summary_table_CE; results_minLS_CE results_numPTS_CE results_numTrees_CE results_min_estce_CE results_total_train_time_CE];


end

mean_minLS_CE = mean(summary_table_CE(:,1));
variance_minLS_CE = var(summary_table_CE(:,1));
mean_numPTS_CE = mean(summary_table_CE(:,2));
variance_numPTS_CE = var(summary_table_CE(:,2));
mean_numTrees_CE = mean(summary_table_CE(:,3));
variance_numTrees_CE = var(summary_table_CE(:,3));
mean_min_estce_CE = mean(summary_table_CE(:,4));
variance_min_estce_CE = var(summary_table_CE(:,4));
mean_total_train_time_CE = mean(summary_table_CE(:,5));
variance_total_train_time_CE = var(summary_table_CE(:,5));


%% Train model using optimal hyperparamater settings learned from Bayesian Optimisation steps on all the training data


% As defined by mean estimated model as defined from 20 x Bayesian Optimisation runs using
% MCR as the error metric with which to assess the objective function


final_minLS_MCR = round(mean_minLS_MCR);
final_numPTS_MCR = round(mean_numPTS_MCR); 
final_numTrees_MCR = round(mean_numTrees_MCR); 
% Generate final trained model from training data
final_mdl_MCR = TreeBagger(final_numTrees_MCR,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',final_minLS_MCR,...
                        'NumPredictorsToSample', final_numPTS_MCR);

order = unique(train_labels); % Order of the group labels                    
% Calculate Confusion matrix
confusion_mat_MCR = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(final_mdl_MCR,...
            test_features...
            )...
            )...
            ),...
            'order', order...
            );
        




% As defined by mean estimated model as defined from 20 x Bayesian Optimisation runs using
% CE as the error metric with which to assess the objective function


final_minLS_CE = round(mean_minLS_CE);
final_numPTS_CE = round(mean_numPTS_CE); 
final_numTrees_CE = round(mean_numTrees_CE); 
% Generate final trained model from training data
final_mdl_CE = TreeBagger(final_numTrees_CE,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',final_minLS_CE,...
                        'NumPredictorsToSample', final_numPTS_CE);
                    
% Calculate Confusion matrix
confusion_mat_CE = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(final_mdl_CE,...
            test_features...
            )...
            )...
            ),...
            'order', order...
            );
        
          
% Draw Confusion matrix                   
confusionchart(confusion_mat_MCR, {'Healthy'; 'Heart_Disease'})
confusionchart(confusion_mat_CE, {'Healthy'; 'Heart_Disease'})


% Initialise figure plots
figure;
hold on;

[recall_MCR, precision_MCR, F1_MCR, specificity_MCR,accuracy_MCR, AUC_MCR] = get_performance(final_mdl_MCR,confusion_mat_MCR, test_features, test_labels);
[recall_CE, precision_CE, F1_CE, specificity_CE,accuracy_CE, AUC_CE] = get_performance(final_mdl_CE,confusion_mat_CE, test_features, test_labels);


  
hold off;



%Function to report model performance
function [recall, precision, F1, specificity,accuracy, AUC] = get_performance(mdl,confusion_mat,test_features, test_labels)

recall = confusion_mat(1)/(confusion_mat(1)+ confusion_mat(3));
precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
F1 = (2*(precision * recall))/(precision + recall);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);

fprintf('Recall %4.2f\n',recall);
fprintf('Precision %4.2f\n',precision);
fprintf('F1 %4.2f\n',F1);
fprintf('Specificity %4.2f\n',specificity);
fprintf('Accuracy %4.2f\n',accuracy);

% Draw ROC curve
[yhat,scores,cost] = predict(mdl,test_features);

%need to find NB method to calculate class scores
%should be able to use predict function

% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(test_labels,scores(:,2), 1);

% Plot ROC curve

plot(fpr,tpr)

xlabel('False Positive Rate')
ylabel('True Positive Rate')


end