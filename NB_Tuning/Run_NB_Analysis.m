addpath(genpath('../'));
[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');
order = unique(train_labels); % Order of the group labels

%% Run grid search and Naive Bayes, testing normal and kernel distributions
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

%% Take the model parameters identified by the CE metric and establish
% performance for these models using accuracy as a metric on the test set

%BayesOpt
CNBMdl_CE_BO = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',44.651);
confusion_mat_CE_BO = confusionmat(test_labels,predict(CNBMdl_CE_BO,test_features),'Order', order);
fprintf('Best cross entropy trained model using Bayesopt for parameter search\n');
get_performance(CNBMdl_CE_BO,confusion_mat_CE_BO,test_features, test_labels);

%Grid search
CNBMdl_CE_GS = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',36.337);
confusion_mat_CE_GS = confusionmat(test_labels,predict(CNBMdl_CE_GS,test_features),'Order', order);
fprintf('Best cross entropy trained model using Grid seach for parameter search\n');
get_performance(CNBMdl_CE_GS,confusion_mat_CE_GS,test_features, test_labels);

%manual grid search
distributions_CE_manGS = {'kernel','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
CNBMdl_CE_manGS = fitcnb(train_features,train_labels,'DistributionNames',distributions_CE_manGS);
confusion_mat_CE_manGS = confusionmat(test_labels,predict(CNBMdl_CE_manGS,test_features),'Order', order);
fprintf('Best cross entropy trained model using manual Grid seach for parameter search\n');
get_performance(CNBMdl_CE_manGS,confusion_mat_CE_manGS,test_features, test_labels);

%The manual grid search gives significantly better performance. Test optimising
%the kernel width to see if this gives any additional improvement

hpOO3 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');
CVNBMdl_CE_manGS_optwidth = fitcnb(train_features,train_labels,'DistributionNames',distributions_CE_manGS, ...
    'OptimizeHyperparameters',{'Width'},'HyperparameterOptimizationOptions',hpOO3);
confusion_mat_CE_manGS_optwidth = confusionmat(test_labels,predict(CVNBMdl_CE_manGS_optwidth,test_features),'Order', order);
fprintf('Best cross entropy trained model using manual grid seach with kernel width optimisation for parameter search\n');
get_performance(CVNBMdl_CE_manGS_optwidth,confusion_mat_CE_manGS_optwidth,test_features, test_labels);

%% Now compare this to doing the same thing using MCR

%BayesOpt
CNBMdl_MCR_BO = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',30.958);
confusion_mat_MCR_BO = confusionmat(test_labels,predict(CNBMdl_MCR_BO,test_features),'Order', order);
fprintf('Best MCR trained model using Bayesopt for parameter search\n');
get_performance(CNBMdl_MCR_BO,confusion_mat_MCR_BO,test_features, test_labels);
 
%Grid search
CNBMdl_MCR_GS = fitcnb(train_features,train_labels,'DistributionNames','kernel','Width',36.337);
confusion_mat_MCR_GS = confusionmat(test_labels,predict(CNBMdl_MCR_GS,test_features),'Order', order);
fprintf('Best MCR trained model using grid seach for parameter search\n');
get_performance(CNBMdl_MCR_GS,confusion_mat_MCR_GS,test_features, test_labels);
 
%manual grid search
distributions_MCR_manGS = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
CNBMdl_MCR_manGS = fitcnb(train_features,train_labels,'DistributionNames',distributions_MCR_manGS);
confusion_mat_MCR_manGS = confusionmat(test_labels,predict(CNBMdl_MCR_manGS,test_features),'Order', order);
fprintf('Best MCR trained model using manual grid seach for parameter search\n');
get_performance(CNBMdl_MCR_manGS,confusion_mat_MCR_manGS,test_features, test_labels);
 
%Test optimising the kernel width to see if this gives any additional improvement
 
hpOO3 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');
CVNBMdl_MCR_manGS_optwidth = fitcnb(train_features,train_labels,'DistributionNames',distributions_MCR_manGS, ...
    'OptimizeHyperparameters',{'Width'},'HyperparameterOptimizationOptions',hpOO3);
confusion_mat_MCR_manGS_optwidth = confusionmat(test_labels,predict(CVNBMdl_MCR_manGS_optwidth,test_features),'Order', order);
fprintf('Best MCR trained model using manual grid seach with kernel width optimisation for parameter search\n');
get_performance(CVNBMdl_MCR_manGS_optwidth,confusion_mat_MCR_manGS_optwidth,test_features, test_labels);


%% Select final best model

%Test the time to run optimise the kernel width 
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};
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
get_performance(CNBMdl_final,confusion_mat,train_features,train_labels);

