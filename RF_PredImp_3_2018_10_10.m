[In, Infemale, Inmale, Out, Outfemale, Outmale] = loadheart('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv'); % see script file loadhear.m

par = devicespec(); % see script file devicespec.m

%data partition
cp = cvpartition(Out,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 


% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_matrix = []; % Initialise matrix

for i = [1:cp.NumTestSets]

% Generate predictor importance values for each fold
XTRAIN = In([training(cp,i)],:);
YTRAIN = Out([training(cp,i)]);

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


%% Female tables

cp_female = cvpartition(Outfemale,'KFold',10);

% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_female_matrix = []; % Initialise matrix

for i = [1:cp_female.NumTestSets]

% Generate predictor importance values for each fold
XTRAIN_female = Infemale([training(cp_female,i)],:);
YTRAIN_female = Outfemale([training(cp_female,i)]);

RFmdl_female = TreeBagger(50,XTRAIN_female,YTRAIN_female,...
           'Method','classification',...
           'OOBVarImp','On',...
           'Options',par,...
           'PredictorSelection','curvature',...
           'OOBPredictorImportance','on');
PredImp_female = RFmdl_female.OOBPermutedPredictorDeltaError;

PredImp_female_matrix = [PredImp_female_matrix; PredImp_female]; 

end


PredImp_female_means = mean(PredImp_female_matrix, 1); % Calculate mean of each column

% Generate bar chart show Mean Predictor Importance values for all 13
% Predictor Vaiables
figure;
bar(PredImp_female_means)
title("Importance of each Predictor Variable - Females", 'fontsize',22);
ylabel('Predictor importance estimates', 'fontsize',16);
xlabel('Predictors', 'fontsize',16);
h = gca;
h.XTickLabel = RFmdl_female.PredictorNames;
h.FontSize = 14;


% Remove predictor variables from table In which have a mean Predictor
% Importance <0.4

% In_high_imp_variables = removevars(In,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});



%% Male tables
cp_male = cvpartition(Outmale,'KFold',10);

% Calculate mean prediction importance value for each predictor over
% 10-fold CV data
PredImp_male_matrix = []; % Initialise matrix

for i = [1:cp_male.NumTestSets]

% Generate predictor importance values for each fold
XTRAIN_male = Inmale([training(cp_male,i)],:);
YTRAIN_male = Outmale([training(cp_male,i)]);

RFmdl_male = TreeBagger(50,XTRAIN_male,YTRAIN_male,...
           'Method','classification',...
           'OOBVarImp','On',...
           'Options',par,...
           'PredictorSelection','interaction-curvature',...
           'OOBPredictorImportance','on');
PredImp_male = RFmdl_male.OOBPermutedPredictorDeltaError;

PredImp_male_matrix = [PredImp_male_matrix; PredImp_male]; 

end


PredImp_male_means = mean(PredImp_male_matrix, 1); % Calculate mean of each column

% Generate bar chart show Mean Predictor Importance values for all 13
% Predictor Vaiables
figure;
bar(PredImp_male_means)
title("Importance of each Predictor Variable - Males", 'fontsize',22);
ylabel('Predictor importance estimates', 'fontsize',16);
xlabel('Predictors', 'fontsize',16);
h = gca;
h.XTickLabel = RFmdl_male.PredictorNames;
h.FontSize = 14;


% Remove predictor variables from table In which have a mean Predictor
% Importance <0.4

% In_high_imp_variables = removevars(In,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});