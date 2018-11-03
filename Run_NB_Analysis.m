[train_features, train_labels, test_features, test_labels, X_header, cp] ...
    = load_heart_csv('heart.csv','numeric','array');

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


