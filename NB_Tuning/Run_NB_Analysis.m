addpath(genpath('../'));
[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');

%Run grid search and Naive Bayes, testing normal and kernel distributions
%on the features and optimising the kernel width

%Suppress warnings about standardising data before using kernel width
%feature, we are doing this below
warning('off','stats:bayesoptim:bayesoptim:StandardizeIfOptimizingNBKernelWidth');
%Suppress warnings about not identifying categorical data before using
%Mvmn, matlab will add relevant feature to the categorical factors list
warning('off','stats:ClassificationNaiveBayes:ClassificationNaiveBayes:SomeMvmnNotCat');

fprintf('Running optimisation of features using kernel and normal distributions for all features\n')
Naive_Bayes_optimisation(train_features,train_labels,cp)
fprintf('Running grid search of features using kernel and normal distributions for each feature\n')
Naive_Bayes_man_gs(train_features,train_labels,X_header,cp)

train_features_std = zscore(train_features);

fprintf('Running optimisation of features using kernel and normal distributions for all features with standardisation\n')
Naive_Bayes_optimisation(train_features_std,train_labels,cp)
fprintf('Running grid search of features using kernel and normal distributions for each feature with standardisation\n')
Naive_Bayes_man_gs(train_features_std,train_labels,X_header,cp)


%Take the model parameters identified by the CE metric and compute accuracy to see
%if CE provides a better training metric

%BayesOpt model
CNBMdl_CE_BO = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',44.651);
confusion_mat_CE_BO = confusionmat(test_labels,predict(CNBMdl_final,test_features),'Order', order);
fprintf('Best cross entropy trained model using Bayesopt for parameter search\n');
get_performance(CNBMdl_CE_BO,confusion_mat_CE_BO,train_features, train_labels);

%Grid search
CNBMdl_CE_GS = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',36.337);
confusion_mat_CE_GS = confusionmat(test_labels,predict(CNBMdl_final,test_features),'Order', order);
fprintf('Best cross entropy trained model using Grid seach for parameter search\n');
get_performance(CNBMdl_CE_GS,confusion_mat_CE_GS,train_features, train_labels);

%manual grid search
distributions_CE_manGS = {'kernel','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
CNBMdl_CE_manGS = fitcnb(train_features,train_labels,'DistributionNames',distributions_CE_manGS);
confusion_mat_CE_manGS = confusionmat(test_labels,predict(CNBMdl_final,test_features),'Order', order);
fprintf('Best cross entropy trained model using manual Grid seach for parameter search\n');
get_performance(CNBMdl_CE_manGS,confusion_mat_CE_manGS,train_features, train_labels);


%The best accuracy we can achieve using cross entropy as a training metric
%is less than using MCR as a training metric. The above models all achieve
%78.85% accuracy compared to 84.43% accuracy from using missclassification rate
%as the training metric

%Test optimising the kernel width to see if this gives any additional
%improvement for the best result from the grid search using MCR as metric
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
hpOO3 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

tic;
CVNBMdl3 = fitcnb(train_features,train_labels,'DistributionNames',distributions, ...
    'OptimizeHyperparameters',{'Width'},'HyperparameterOptimizationOptions',hpOO3);
optimise_width_time = toc;


%Run final model on test data using the distributions and width identified
%above

tic;
CNBMdl_final = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);
final_model_train_time = toc;

order = unique(train_labels); % Order of the group labels

tic;
confusion_mat = confusionmat(test_labels,predict(CNBMdl_final,test_features),'Order', order);
final_model_test_time = toc;

% Draw Confusion matrix 
% Check for dependencies
v = ver();
if any(strcmp('Deep Learning Toolbox',{v.Name}))
    confusionchart(confusion_mat, {'Healthy'; 'Heart_Disease'})
end

%Plot ROC curve for NB final model
get_performance(CNBMdl_final,confusion_mat,train_features, train_labels);

