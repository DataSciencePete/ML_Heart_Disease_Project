%  Load data into Matlab
addpath(genpath('../'));
[train_features, train_labels, test_features, test_labels, X_header,cvp] = load_heart_csv('Data/heart.csv', 'table', 'categorical'); % see script file loadhear.m

% Check for dependencies
v = ver();
if any(strcmp('Parallel Computing Toolbox',{v.Name}))
    par = devicespec(); % see script file devicespec.m
else
    fprintf('Skipping device specification, parallel computing toolbox not installed\n');
    par = statset('UseParallel',false);
end

%data partition
cp = cvpartition(train_labels,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 


% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_matrix = []; % Initialise matrix

for i = [1:cp.NumTestSets]

% Generate predictor importance values for each fold
XTRAIN = train_features([training(cp,i)],:);
YTRAIN = train_labels([training(cp,i)]);

RFmdl = TreeBagger(50,XTRAIN,YTRAIN,...
           'Method','classification',...
           'OOBVarImp','On',...
           'Options',par,...
           'PredictorSelection','curvature',...
           'OOBPredictorImportance','on');
PredImp = RFmdl.OOBPermutedPredictorDeltaError;

PredImp_matrix = [PredImp_matrix; PredImp]; 

end


PredImp_means = mean(PredImp_matrix, 1); % Calculate mean of each column

% Generate bar chart show Mean Predictor Importance values for all 13
% Predictor Vaiables
figure;
bar(PredImp_means)
title("Importance of each Predictor Variable", 'fontsize',22);
ylabel('Predictor importance estimates', 'fontsize',16);
xlabel('Predictors', 'fontsize',16);
h = gca;
h.XTickLabel = RFmdl.PredictorNames;
h.FontSize = 14;


% Remove predictor variables from table In which have a mean Predictor
% Importance <0.4

%In_high_imp_variables = removevars(In,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});


% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_matrix = []; % Initialise matrix


