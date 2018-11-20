%  Load data into Matlab
[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');

% Prepare hyperparamter object for optimisation using Bayesian Optimisation

%{
% Data distribution types
f1 = optimizableVariable('f1', {'normal' 'kernel'},'Type','categorical');
f2 = optimizableVariable('f2', {'normal' 'kernel'},'Type','categorical');
f3 = optimizableVariable('f3', {'normal' 'kernel'},'Type','categorical');
f4 = optimizableVariable('f4', {'normal' 'kernel'},'Type','categorical');
f5 = optimizableVariable('f5', {'normal' 'kernel'},'Type','categorical');
%}

% Generate range of widths to explore for BO algorithm used settings
% described in fitcnb source code see BayesoptInfo.m for details
max_predic_range = max(nanmax(train_features,[],2) - nanmin(train_features,[],2));

diffs = diff(sort(train_features, 2), 1, 2);
min_predict_diff = nanmin(diffs(diffs~=0));

dist = optimizableVariable('dist', {'normal' 'kernel'},'Type','categorical');
widthparam = optimizableVariable('widthparam', [min_predict_diff/4, max(max_predic_range, min_predict_diff)]);

%Hyperparamter object
hyperparametersNB = [dist; widthparam];


results_CE = bayesopt(@(params)myCVlossfcn_nbce(params,train_features,train_labels,cp),hyperparametersNB, 'MaxObjectiveEvaluations', 30, 'ExplorationRatio', 0.5, 'Verbose',1);

results_MCR = bayesopt(@(params)myCVlossfcn_nb(params,train_features,train_labels,cp),hyperparametersNB, 'MaxObjectiveEvaluations', 30, 'ExplorationRatio', 0.5, 'Verbose',1);

order = unique(train_labels); % Order of the group labels  

%Build a model from optimum CE parameters

distParam_CE = char(results_CE.XAtMinEstimatedObjective.dist);
widthParam_CE = results_CE.XAtMinEstimatedObjective.widthparam;

NBMdl_CE = fitcnb(train_features,train_labels,...
    'DistributionNames',distParam_CE,'Width',widthParam_CE);            

confusion_mat_CE = confusionmat(test_labels,...
            predict(NBMdl_CE,test_features)...
            ,'Order', order);

%Initialise plots
figure;
hold on;
        
        
[recall_CE, precision_CE, F1_CE, specificity_CE,accuracy_CE, AUC_CE] = get_performance(NBMdl_CE,confusion_mat_CE, train_features, train_labels);

%Build a model from optimum MCR parameters

distParam_MCR = char(results_MCR.XAtMinEstimatedObjective.dist);
widthParam_MCR = results_MCR.XAtMinEstimatedObjective.widthparam;

if strcmp(distParam_MCR, 'kernel')
    NBMdl_MCR = fitcnb(train_features,train_labels,...
        'DistributionNames',distParam_MCR,'Width',widthParam_MCR);            
else
    NBMdl_MCR = fitcnb(train_features,train_labels,...
        'DistributionNames',distParam_MCR);            
end

confusion_mat_MCR = confusionmat(test_labels,...
            predict(NBMdl_MCR,test_features)...
            ,'Order', order);

% Generate performance metrics for training data using generated hyperaparamters        
[recall_MCR, precision_MCR, F1_MCR, specificity_MCR,accuracy_MCR, AUC_MCR] = get_performance(NBMdl_MCR,confusion_mat_MCR, train_features, train_labels);

ax = gca; % grab current axis
ax.FontSize = 16 % Alter font size
ax.FontWeight = 'bold';

% add legend entries
lg = legend('Cross Entropy', 'MCR');

