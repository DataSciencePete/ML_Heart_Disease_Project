[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');

rng(1);

order = unique(train_labels); % Order of the group labels

%Run final Naive Bayes Model
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};

tic;
NB_final_mdl = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);
NB_train_time = toc;

tic;
NB_confusion_mat = confusionmat(test_labels,...
            predict(NB_final_mdl,test_features)...
            ,'Order', order);
NB_predict_test_time = toc;

% Initialise figure plots
figure;
hold on;

get_performance(NB_final_mdl,NB_confusion_mat, test_features, test_labels);

fprintf('Naive Bayes train time %4.2fs\n',NB_train_time);

[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','table','categorical');

order = unique(train_labels); % Order of the group labels(for categorical column)

%Run final Random Forest Model
In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

par = devicespec(); % see script file devicespec.m


tic;
RF_final_mdl = TreeBagger(151,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',30,...
                        'NumPredictorsToSample', 2);
RF_train_time = toc; % Time taken to train final RF model on all the training data

% Confusion matrix on training data
RF_confusion_mat_train = confusionmat(train_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(RF_final_mdl,train_features))),...
            'order', order);
% Confusion matrix on test data
tic;
RF_confusion_mat_test = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(RF_final_mdl,test_features))),...
            'order', order);
 
RF_predict_test_time = toc; % Time taken for optimised model to make test set predictions
        
% Get performance data for training data
[RF_recall_Train, RF_precision_Train, RF_F1_Train, RF_specificity_Train, RF_accuracy_Train, RF_AUC_Train] = get_performance(RF_final_mdl,RF_confusion_mat_train, train_features, train_labels);        
% Get performance data for test data
[RF_recall_Test, RF_precision_Test, RF_F1_Test, RF_specificity_Test, RF_accuracy_Test, RF_AUC_Test] = get_performance(RF_final_mdl,RF_confusion_mat_test, test_features, test_labels);

fprintf('Random Forest train time %4.2fs\n',RF_train_time);
  
hold off;

%Function to report model performance
function [recall, precision, F1, specificity,accuracy, AUC] = get_performance(mdl,confusion_mat,test_features, test_labels)

rng(1);
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


