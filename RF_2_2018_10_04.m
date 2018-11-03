[train_features, train_labels, test_features, test_labels, X_header, cvp] = loadheart('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv'); % see script file loadhear.m
% [train_features, train_labels, test_features, test_labels, X_header] = load_heart_csv('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv', 'numeric', 'table'); % see script file loadhear.m
par = devicespec(); % see script file devicespec.m


%% Section 1
% Set up Cross validation using Random Forest see https://uk.mathworks.com/help/stats/treebagger.html for hyperparameter settings and https://uk.mathworks.com/matlabcentral/answers/34771-classifier for code
tic % time how long it takes process to run     

%data partition
cp = cvpartition(train_labels,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(50,XTRAIN,YTRAIN,...
                                                        'Method','classification',...
                                                        'OOBVarImp','On',...
                                                        'Options',par...
                                                       ),XTEST));
order = unique(train_labels); % Order of the group labels
confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,...
                                                       categorical(...
                                                       cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
                                                       predict(TreeBagger(50,XTRAIN,YTRAIN,...
                                                                          'Method','classification',...
                                                                          'OOBVarImp','On',...
                                                                          'Options',par...
                                                                         ),...
                                                                XTEST...
                                                               ))),...
                                                       'order', order...
                                                      )...
                                          );
                              
% missclassification error 
missclasfError = crossval('mcr',train_features,train_labels,'predfun',classF,'partition',cp);
cfMat = crossval(confusionF,train_features,train_labels,'partition',cp); % Matrix shows number of correctly and incorrectly classified samples for each classification for each of the 10 cross validated data sets
cfMat = reshape(sum(cfMat),2,2); % summation of the 10 confusion matrices over the 10CV data sets
% Generate confusion matrix
confusionchart(cfMat, {'Healthy'; 'Heart_Disease'})

%Calculate recall, precision and F1 score
recall = cfMat(1)/(cfMat(1)+ cfMat(3));
precision = cfMat(1)/(cfMat(1) + cfMat(2));
F1 = 2*(precision * recall)/(precision + recall);


toc


%% Section 2
% Bayesian Optimisation

% Min No of Observations per leaf
maxMinLS = 50;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
% No of variables to consider at each split in the tree

% Remove predictor variables from table In which have a mean Predictor
% Importance <0.4

In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

numPTS = optimizableVariable('numPTS',[1,size(In_high_imp_variables,2)],'Type','integer'); % define number of predictors

maxNumTrees = 500;
numTrees = optimizableVariable('numTrees',[1,maxNumTrees],'Type','integer'); % define max number of trees to define forest

hyperparametersRF = [minLS; numTrees];

% Also also consider tuning the number of trees in the ensemble




result = bayesopt(@(params)Optimisation(params,In_high_imp_variables,train_labels),hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','Verbose',1);



%% Section 3 Bayesian Opimisation with Cross validation
rng(1);
% results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,50),hyperparametersRF, 'AcquisitionFunctionName', 'probability-of-improvement', 'IsObjectiveDeterministic', true, 'MaxObjectiveEvaluations', 30);

% results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par, 50),hyperparametersRF, 'MaxObjectiveEvaluations', 30);
% results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par, 50),hyperparametersRF, 'MaxObjectiveEvaluations', 60);
% results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par, 50),hyperparametersRF, 'MaxObjectiveEvaluations', 120);

% Run using MCR as loss function
results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 20);
results_paralllel = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 20, 'UseParallel',true);

results_paralllel_er0 = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0, 'Verbose',1);
results_paralllel_er0_25 = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0.25, 'Verbose',1);
results_paralllel_er0_5_MCR = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 100, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
results_paralllel_er0_75 = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0.75, 'Verbose',1);
results_paralllel_er1= bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 1, 'Verbose',1);

% Run using 1-F1 as loss function
results_paralllel_er0_5_F1 = bayesopt(@(params)myCVlossfcn_F1(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 100, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);


% Run bayesopt 20 times and calculate the mean hyperparamter values for RF
% Initialise summary table
summary_table = [];
for i = [1:20]
    % Rerun bayesian optimisation 20 times to generate a set of values from
    % which an average hyperparameter setting can be gleaned (numTrees set
    % at 100 
    results_paralllel_er0_5_MCR = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 100, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    results_minLS = results_paralllel_er0_5_MCR.XAtMinEstimatedObjective.minLS;
    results_numPTS = results_paralllel_er0_5_MCR.XAtMinEstimatedObjective.numPTS;
    results_min_est_mcr = results_paralllel_er0_5_MCR.MinEstimatedObjective;
    results_total_train_time = results_paralllel_er0_5_MCR.TotalElapsedTime;
    summary_table = [summary_table; results_minLS results_numPTS results_min_est_mcr results_total_train_time];


end

mean_minLS = mean(summary_table(:,1));
mean_numPTS = mean(summary_table(:,2));
mean_min_est_mcr = mean(summary_table(:,3));
mean_total_train_time = mean(summary_table(:,4));


% Run bayesopt 20 times and calculate the mean hyperparamter values for RF
% numTrees
% Initialise summary table
summary_table_numTrees_minLS = [];
for i = [1:20]
    % Rerun bayesian optimisation 20 times to generate a set of values from
    % which an average hyperparameter setting can be gleaned (numTrees set
    % at 100 
    results_paralllel_er0_5_MCR_2 = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 100, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    results_minLS_2 = results_paralllel_er0_5_MCR_2.XAtMinEstimatedObjective.minLS;
    results_numTrees = results_paralllel_er0_5_MCR_2.XAtMinEstimatedObjective.numTrees;
    results_min_est_mcr_2 = results_paralllel_er0_5_MCR_2.MinEstimatedObjective;
    results_total_train_time_2 = results_paralllel_er0_5_MCR_2.TotalElapsedTime;
    summary_table_numTrees_minLS = [summary_table_numTrees_minLS; results_minLS_2 results_numTrees results_min_est_mcr_2 results_total_train_time_2];


end

mean_minLS_2 = mean(summary_table_numTrees_minLS(:,1));
mean_numTrees = mean(summary_table_numTrees_minLS(:,2));
mean_min_est_mcr_2 = mean(summary_table_numTrees_minLS(:,3));
mean_total_train_time_2 = mean(summary_table_numTrees_minLS(:,4));
%% 
rng(1);
% As defined by mean estimated model as defined from 20 x Bayesian
% Optimisation runs

final_numPTS = round(mean_numPTS);
final_minLS = round(mean_minLS);
final_numTrees = round(mean_numTrees);
% Generate final trained model from training data
final_mdl = TreeBagger(final_numTrees,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',final_minLS,...
                        'NumPredictorsToSample', final_numPTS);
                    
% Calculate Confusion matrix
confusion_mat = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(final_mdl,...
            test_features...
            )...
            )...
            ),...
            'order', order...
            );
        
% Draw Confusion matrix                   
confusionchart(confusion_mat, {'Healthy'; 'Heart_Disease'})

%Calculate recall, precision and F1 score
recall = confusion_mat(1)/(confusion_mat(1)+ confusion_mat(3));
precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
F1 = (2*(precision * recall))/(precision + recall);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);



% Draw ROC curve
% Calc OOB classes and associated OOB predicted probabilities that each
% individual is healthy Sfit(:,1) or has heart disease Sfit(:,2)
[Yfit,Sfit] = oobPredict(final_mdl)
% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(final_mdl.Y,Sfit(:,2), 1)

% Plot ROC curve
figure
plot(fpr,tpr)
xlabel('False Positive Rate')
ylabel('True Positive Rate')

%% Section 4 Grid Search Random Forest approach

minLS_grid = linspace(1,20,20);  % Min No of observations per leaf (paramter search space)
numPTS_grid = linspace(1,size(train_features,2),size(train_features,2)); % Number of variables to select at random for each decision split (paramter search space)

[LS,P] = ndgrid(minLS_grid, numPTS_grid); % Parameter grid

fitresult = arrayfun(@(p1,p2) fittingfunction(p1,p2), F, S); %run a fitting on every pair fittingfunction(F(J,K), S(J,K))
result_grid = arrayfun(@(l,p)myCVlossfcn_grid(l,p,train_features,train_labels,par), LS, P);
%%
% testmodel = TreeBagger(50,train_features,train_labels,...
%           'Method','classification',...
%           'OOBVarImp','On',...
%           'Options',par,...
%           'OOBPredictorImportance','on'...
%           )
% checkpredict = predict(testmodel, In)
% 
% checkpredict = cellfun(@str2num, checkpredict);
% confusionmat(train_labels, checkpredict)