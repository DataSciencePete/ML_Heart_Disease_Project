
[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','numeric','array');

%%

%Run grid search and Naive Bayes, testing normal and kernel distributions
%on the features and optimising the kernel width


%Suppress warnings about standardising data before using kernel width
%feature, we are doing this below
warning('off','stats:bayesoptim:bayesoptim:StandardizeIfOptimizingNBKernelWidth');

fprintf('Running optimisation of features using kernel and normal distributions for all features')
Naive_Bayes_optimisation(train_features,train_labels,cp)
fprintf('Running grid search of features using kernel and normal distributions for each feature')
Naive_Bayes_man_gs(train_features,train_labels,X_header,cp)

train_features = zscore(train_features);

fprintf('Running optimisation of features using kernel and normal distributions for all features with standardisation')
Naive_Bayes_optimisation(train_features,train_labels,cp)
fprintf('Running grid search of features using kernel and normal distributions for each feature with standardisation')
Naive_Bayes_man_gs(train_features,train_labels,X_header,cp)

%%

%Test optimising the kernel width to see if this gives any additional
%improvement for the best result from the grid search
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
hpOO3 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

CVNBMdl3 = fitcnb(train_features,train_labels,'DistributionNames',distributions, ...
    'OptimizeHyperparameters',{'Width'},'HyperparameterOptimizationOptions',hpOO3);


%Run final model on test data using the distributions and width identified
%above

CNBMdl_final = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);

order = unique(train_labels); % Order of the group labels
confusion_mat = confusionmat(test_labels,predict(CNBMdl_final,test_features),'Order', order);

% Draw Confusion matrix                   
%%confusionchart(confusion_mat, {'Healthy'; 'Heart_Disease'})

%Calculate recall, precision and F1 score
recall = confusion_mat(1)/(confusion_mat(1)+ confusion_mat(3));
precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
F1 = (2*(precision * recall))/(precision + recall);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);


%%

% Draw ROC curve
[yhat,scores,cost] = predict(CNBMdl_final,test_features);

%need to find NB method to calculate class scores
%should be able to use predict function

% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(test_labels,scores(:,2), 1);

% Plot ROC curve
figure
plot(fpr,tpr)
xlabel('False Positive Rate')
ylabel('True Positive Rate')


