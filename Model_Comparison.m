[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');

order = unique(train_labels); % Order of the group labels

%Run final Naive Bayes Model
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};

tic;
NB_final_mdl = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);
NB_train_time = toc;

NB_confusion_mat = confusionmat(test_labels,...
            predict(NB_final_mdl,test_features)...
            ,'Order', order);

% Initialise figure plots
figure;
hold on;

get_performance(NB_final_mdl,NB_confusion_mat, test_features, test_labels);

fprintf('Naive Bayes train time %4.2fs\n',NB_train_time);

[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','table','categorical');

order = unique(train_labels); % Order of the group labels(for categorical column)

%Run final Random Forest Model
In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

tic;
RF_final_mdl = TreeBagger(109,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'MinLeafSize',29,...
                        'NumPredictorsToSample', 1);
               
RF_train_time = toc;

RF_confusion_mat = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(RF_final_mdl,test_features))),...
            'order', order);
       
get_performance(RF_final_mdl,RF_confusion_mat, test_features, test_labels);

fprintf('Random Forest train time %4.2fs\n',RF_train_time);
  
hold off;

%Function to report model performance
function get_performance(mdl,confusion_mat,test_features, test_labels)

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


