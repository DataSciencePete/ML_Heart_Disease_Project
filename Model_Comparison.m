[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv');

order = unique(train_labels); % Order of the group labels

%Run final Naive Bayes Model
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};

tic;
NB_final_mdl = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);
NB_train_time = toc;

NB_confusion_mat = confusionmat(test_labels,...
            predict(CNBMdl_final,test_features)...
            ,'Order', order);

get_performance(NB_final_mdl,NB_confusion_mat);
get_performance(RF_final_mdl,RF_confusion_mat);
fprintf('Naive Bayes train time %4.2fs\n',NB_train_time);

[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','table');

%Run final Random Forest Model
In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

tic;
RF_final_mdl = TreeBagger(109,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'Options',par,...
                        'MinLeafSize',29,...
                        'NumPredictorsToSample', 1);
rf_train_time = toc;
fprintf('Random Forest train time %4.2fs\n',RF_train_time);
                    
RF_confusion_mat = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(final_mdl,test_features))),...
            'order', order);
                   



%Function to report model performance
function get_performance(mdl,confusion_mat)

recall = confusion_mat(1)/(confusion_mat(1)+ confusion_mat(3));
precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
F1 = (2*(precision * recall))/(precision + recall);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);

print('Recall %4.2f\n',recall);
print('Precision %4.2f\n',precision);
print('F1 %4.2f\n',F1);
print('Specificity %4.2f\n',specificity);
print('Accuracy %4.2f\n',accuracy);

% Draw ROC curve
[yhat,scores,cost] = predict(mdl,test_features);

%need to find NB method to calculate class scores
%should be able to use predict function

% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(test_labels,scores(:,2), 1);

% Plot ROC curve
figure
plot(fpr,tpr)
hold
xlabel('False Positive Rate')
ylabel('True Positive Rate')

end


