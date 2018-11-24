%% Load data into Matlab
% Add folders to the path
addpath(genpath('../'));
[train_features, train_labels, test_features, test_labels, X_header, cvp] = load_heart_csv('heart.csv', 'table', 'categorical');

% Check for dependencies
v = ver();
if any(strcmp('Parallel Computing Toolbox',{v.Name}))
    par = devicespec(); % see script file devicespec.m
else
    fprintf('Skipping device specification, parallel computing toolbox not installed\n');
    par = statset('UseParallel',false);
end
%%
% Create data cross validation partition object
% Create 10-folds cross-validation partition for data.
% Each subsample has roughly equal size and roughly the same class
% proportions as in original data set
cp = cvpartition(train_labels,'KFold',10);


% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_matrix = []; % Initialise matrix

for i = 1:cp.NumTestSets

% Generate predictor importance values for each fold
XTRAIN = train_features([training(cp,i)],:);
YTRAIN = train_labels([training(cp,i)]);

% Generate RF model for each k-fold partitions
RFmdl = TreeBagger(50,XTRAIN,YTRAIN,...
           'Method','classification',...
           'OOBVarImp','On',...
           'Options',par,...
           'PredictorSelection','curvature',...
           'OOBPredictorImportance','on');
% Calculate predictor importance for each feature
PredImp = RFmdl.OOBPermutedPredictorDeltaError;
% Populate output matrix
PredImp_matrix = [PredImp_matrix; PredImp];

end

% Calculate mean predictor importance for each feature over all 10 CV folds
PredImp_means = mean(PredImp_matrix, 1);

% Generate bar chart show Mean Predictor Importance values for all 13
% Predictor Vaiables
figure;
bar(PredImp_means)
title("Importance of each Predictor Variable", 'fontsize',22);
ylabel('Predictor importance estimates', 'fontsize',16);
xlabel('Predictors', 'fontsize',16);
ax = gca; % grab current axes
ax.XTickLabel = RFmdl.PredictorNames;
ax.FontSize = 16;
ax.FontWeight = 'bold';
ax.XTickLabelRotation = 45;


