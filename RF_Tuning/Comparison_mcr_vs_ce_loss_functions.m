%% Section 1
%  Load data into Matlab
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

%% Section 3 MCR as loss function - Run Bayesian Optimisation using MCR as error value to evaluate objective 20 times and generate 20 calculations for numTrees, minLS and numPTS
%  Calculate mean and variance values for training accuracy, recall,
%  precision, F1, Specificity and AUC

% Run bayesopt using MCR as error value to evaluate objective function 20 times and calculate 20 hyperparameter calculations for numTrees, minLS and numPTS
% Initialise summary table to be populated with performance metrics
summary_table_MCR = [];
best_accuracy_MCR = 0;
for i = 1:20
    
    results_paralllel_MCR = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    minLS_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.minLS;
    numPTS_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.numPTS;
    numTrees_MCR = results_paralllel_MCR.XAtMinEstimatedObjective.numTrees;
    
    % Train model using optimal hyperparamater settings learned from Bayesian Optimisation steps on all the training data
 
    % Generate trained model from all training data
    mdl_MCR = TreeBagger(numTrees_MCR,In_high_imp_variables, train_labels,...
                            'method','classification',...
                            'OOBPrediction','on',...
                            'Options',par,...
                            'MinLeafSize',minLS_MCR,...
                            'NumPredictorsToSample', numPTS_MCR);

    order = unique(train_labels); % Order of the group labels                    
    % Calculate Confusion matrix from which performance metrics are
    % calulated
    confusion_mat_MCR = confusionmat(train_labels,...
                categorical(...
                cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
                predict(mdl_MCR,...
                train_features...
                )...
                )...
                ),...
                'order', order...
                );
     
    % Generate performance metrics for training data using generated hyperaparamters        
    [recall_MCR, precision_MCR, F1_MCR, specificity_MCR,accuracy_MCR, AUC_MCR] = get_performance(mdl_MCR,confusion_mat_MCR, train_features, train_labels);
    % Add performance metrics for hyperparamters generated from the current
    % iteration
    
    if accuracy_MCR > best_accuracy_MCR
        % Obtain hyperparameter settings for highest accuracy model
        best_accuracy_MCR = accuracy_MCR;
        best_minLS_MCR = minLS_CE;
        best_numPTS_MCR = numPTS_MCR;
        best_numTrees_MCR = numTrees_MCR;
            
    end
    summary_table_MCR = [summary_table_MCR; recall_MCR precision_MCR F1_MCR specificity_MCR accuracy_MCR AUC_MCR];

    
end

% Calculate mean and SD values for all performance metrics calulated using
% CE as the loss function
mean_recall_MCR = mean(summary_table_MCR(:,1));
variance_recall_MCR = sqrt(var(summary_table_MCR(:,1)));
mean_precision_MCR = mean(summary_table_MCR(:,2));
variance_precision_MCR = sqrt(var(summary_table_MCR(:,2)));
mean_F1_MCR = mean(summary_table_MCR(:,3));
variance_F1_MCR = sqrt(var(summary_table_MCR(:,3)));
mean_specificity_MCR = mean(summary_table_MCR(:,4));
variance_specificity_MCR = sqrt(var(summary_table_MCR(:,4)));
mean_accuracy_MCR = mean(summary_table_MCR(:,5));
variance_accuracy_MCR = sqrt(var(summary_table_MCR(:,5)));
mean_AUC_MCR = mean(summary_table_MCR(:,6));
variance_AUC_MCR = sqrt(var(summary_table_MCR(:,6)));

%% Section 4 CE as loss function - Run Bayesian Optimisation using ce as error value to evaluate objective 20 times and generate 20 calculations for numTrees, minLS and numPTS
%  Calculate mean and variance values for training accuracy, recall,
%  precision, F1, Specificity and AUC

% Run bayesopt using ce as error value to evaluate objective function 20 times and calculate 20 hyperparameter calculations for numTrees, minLS and numPTS
% Initialise summary table to be populated with performance metrics
summary_table_CE = [];
best_accuracy_CE = 0;

for i = 1:20
    
    results_paralllel_CE = bayesopt(@(params)myCVlossfcn_ce(params,In_high_imp_variables,train_labels,par,cvp),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'UseParallel',true, 'ExplorationRatio', 0.5, 'Verbose',1);
    minLS_CE = results_paralllel_CE.XAtMinEstimatedObjective.minLS;
    numPTS_CE = results_paralllel_CE.XAtMinEstimatedObjective.numPTS;
    numTrees_CE = results_paralllel_CE.XAtMinEstimatedObjective.numTrees;
    
    % Train model using optimal hyperparamater settings learned from Bayesian Optimisation steps on all the training data
 
    % Generate trained model from all training data
    mdl_CE = TreeBagger(numTrees_CE,In_high_imp_variables, train_labels,...
                            'method','classification',...
                            'OOBPrediction','on',...
                            'Options',par,...
                            'MinLeafSize',minLS_CE,...
                            'NumPredictorsToSample', numPTS_CE);

    order = unique(train_labels); % Order of the group labels                    
    % Calculate Confusion matrix from which performance metrics are
    % calulated
    confusion_mat_CE = confusionmat(train_labels,...
                categorical(...
                cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
                predict(mdl_CE,...
                train_features...
                )...
                )...
                ),...
                'order', order...
                );
     
    % Generate performance metrics for training data using generated hyperaparamters        
    [recall_CE, precision_CE, F1_CE, specificity_CE,accuracy_CE, AUC_CE] = get_performance(mdl_CE,confusion_mat_CE, train_features, train_labels);
    % Add performance metrics for hyperparamters generated from the current
    % iteration
    if accuracy_CE > best_accuracy_CE
        % Obtain hyperparameter settings for highest accuracy model
        best_accuracy_CE = accuracy_CE;
        best_minLS_CE = minLS_CE;
        best_numPTS_CE = numPTS_CE;
        best_numTrees_CE = numTrees_CE;
            
    end
    summary_table_CE = [summary_table_CE; recall_CE precision_CE F1_CE specificity_CE accuracy_CE AUC_CE];

    
end

% Calculate mean and SD values for all performance metrics calulated using
% CE as the loss function
mean_recall_CE = mean(summary_table_CE(:,1));
variance_recall_CE = sqrt(var(summary_table_CE(:,1)));
mean_precision_CE = mean(summary_table_CE(:,2));
variance_precision_CE = sqrt(var(summary_table_CE(:,2)));
mean_F1_CE = mean(summary_table_CE(:,3));
variance_F1_CE = sqrt(var(summary_table_CE(:,3)));
mean_specificity_CE = mean(summary_table_CE(:,4));
variance_specificity_CE = sqrt(var(summary_table_CE(:,4)));
mean_accuracy_CE = mean(summary_table_CE(:,5));
variance_accuracy_CE = sqrt(var(summary_table_CE(:,5)));
mean_AUC_CE = mean(summary_table_CE(:,6));
variance_AUC_CE = sqrt(var(summary_table_CE(:,6)));

%% Summary table of mean and sd values for performance metrics calculated using MCR vs CE
summary_matrix_mean_sd_performance_metrics = [];
summary_matrix_mean_sd_performance_metrics = [mean_recall_MCR variance_recall_MCR mean_precision_MCR variance_precision_MCR mean_F1_MCR variance_F1_MCR mean_specificity_MCR variance_specificity_MCR mean_accuracy_MCR variance_accuracy_MCR mean_AUC_MCR variance_AUC_MCR best_minLS_MCR best_numPTS_MCR best_numTrees_MCR];
summary_matrix_mean_sd_performance_metrics = [summary_matrix_mean_sd_performance_metrics; mean_recall_CE variance_recall_CE mean_precision_CE variance_precision_CE mean_F1_CE variance_F1_CE mean_specificity_CE variance_specificity_CE mean_accuracy_CE variance_accuracy_CE mean_AUC_CE variance_AUC_CE best_minLS_CE best_numPTS_CE best_numTrees_CE];

summary_table_mean_sd_performance_metrics = array2table(summary_matrix_mean_sd_performance_metrics, 'VariableNames',{'mean_recall' 'variance_recall' 'mean_precision' 'variance_precision' 'mean_F1' 'variance_F1' 'mean_specificity' 'variance_specificity' 'mean_accuracy' 'variance_accuracy' 'mean_AUC' 'variance_AUC' 'best_minLS' 'best_numPTS' 'best_numTrees'}, 'RowNames', {'MCR', 'Croos Entropy'});

% Draw output bar charts for mean performance metrics on the training data

bar_chart = barh(summary_matrix_mean_sd_performance_metrics(:,1:2:12)');
xlim([0.7,1]) % set y axis limits
ax = gca; % grab handle to current axis
% Add labels to each x tcik
ax.YTickLabel = {'Recall', 'Precision', 'F1', 'Specificity', 'Accuracy', 'AUC'};
legendvals = {'Cross Entropy'; 'MCR'}; % Set legend names
% change order of legend entries
lg = legend([bar_chart(2), bar_chart(1)], legendvals, 'Location', 'southeast');
ax.FontSize = 16 % Alter font size
ax.FontWeight = 'bold';


% Draw ROC curves for best performing RF models on the training data derived using Cross entropy and MCR 
% Generate trained model from all training data
best_mdl_MCR = TreeBagger(best_numTrees_MCR,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',best_minLS_MCR,...
                        'NumPredictorsToSample', best_numPTS_MCR);

order = unique(train_labels); % Order of the group labels                    
% Calculate Confusion matrix from which performance metrics are
% calulated
best_confusion_mat_MCR = confusionmat(train_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(best_mdl_MCR,...
            train_features...
            )...
            )...
            ),...
            'order', order...
            );

% Generate performance metrics for training data using generated hyperaparamters        
[best_recall_MCR, best_precision_MCR, best_F1_MCR, best_specificity_MCR, best_accuracy_MCR, best_AUC_MCR] = get_performance(best_mdl_MCR,best_confusion_mat_MCR, train_features, train_labels);


% Draw ROC curves for best performing RF models on the training data derived using Cross entropy and MCR 
% Generate trained model from all training data
best_mdl_CE = TreeBagger(best_numTrees_CE,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',best_minLS_CE,...
                        'NumPredictorsToSample', best_numPTS_CE);

order = unique(train_labels); % Order of the group labels                    
% Calculate Confusion matrix from which performance metrics are
% calulated
best_confusion_mat_CE = confusionmat(train_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(best_mdl_CE,...
            train_features...
            )...
            )...
            ),...
            'order', order...
            );

% Generate performance metrics for training data using generated hyperaparamters 
hold on;
[best_recall_CE, best_precision_CE, best_F1_CE, best_specificity_CE, best_accuracy_CE, best_AUC_CE] = get_performance(best_mdl_CE,best_confusion_mat_CE, train_features, train_labels);
ax = gca; % grab current axis
ax.FontSize = 16 % Alter font size
ax.FontWeight = 'bold';
hold off;
%%
%Function to report model performance
function [recall, precision, F1, specificity,accuracy, AUC] = get_performance(mdl,confusion_mat,train_features, train_labels)

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
[yhat,scores,cost] = predict(mdl,train_features);

%need to find NB method to calculate class scores
%should be able to use predict function

% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(train_labels,scores(:,2), 1);

% Plot ROC curve

plot(fpr,tpr, 'LineWidth',2)

xlabel('False Positive Rate')
ylabel('True Positive Rate')

% change order of legend entries
lg = legend('Cross Entropy', 'MCR');


end