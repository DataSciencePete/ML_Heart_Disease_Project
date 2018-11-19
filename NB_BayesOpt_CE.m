%% Section 1
%  Load data into Matlab
[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');
% [train_features, train_labels, test_features, test_labels, X_header,cvp] = load_heart_csv('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv', 'table', 'categorical'); % see script file loadhear.m


%% Section 2
% Prepare hyperparamter object for optimisation using Bayesian Optimisation

% Data distribution types
f1 = optimizableVariable('f1', {'normal' 'kernel'},'Type','categorical');
f2 = optimizableVariable('f2', {'normal' 'kernel'},'Type','categorical');
f3 = optimizableVariable('f3', {'normal' 'kernel'},'Type','categorical');
f4 = optimizableVariable('f4', {'normal' 'kernel'},'Type','categorical');
f5 = optimizableVariable('f5', {'normal' 'kernel'},'Type','categorical');
dist = optimizableVariable('dist', {'normal' 'kernel'},'Type','categorical');

% Generate range of widths to explore for BO algorithm used settings
% described in fitcnb source code see BayesoptInfo.m for details
max_predic_range = max(nanmax(train_features,[],2) - nanmin(train_features,[],2));

diffs = diff(sort(train_features, 2), 1, 2);
min_predict_diff = nanmin(diffs(diffs~=0));



widthparam = optimizableVariable('widthparam', [min_predict_diff/4, max(max_predic_range, min_predict_diff)],...
                'Transform', 'log');
kernel_smoother_type = optimizableVariable('Kernel', {'normal', 'box', 'epanechnikov', 'triangle'}, ...
                'Optimize', false);


%Hyperparamter object
hyperparametersRF = [dist; widthparam];

results_paralllel_MCR = bayesopt(@(params)myCVlossfcn_nbce(params,train_features,train_labels,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'AcquisitionFunctionName', 'expected-improvement-plus','UseParallel',true, 'Verbose',2, 'AlwaysReportObjectiveErrors', true);

best_point = bestPoint(results_paralllel_MCR);