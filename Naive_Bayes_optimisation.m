[train_features, train_labels, test_features, test_labels, ...
    X_header, cp] = load_heart_csv('heart.csv','numeric','array');

 
%{

%Set up Cross validation
tic % time how long it takes process to run

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN),XTEST));

order = unique(y); % Order of the group labels

%Confusion function passed to the crossval function
confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,predict(fitcnb(XTRAIN,YTRAIN),XTEST),'order', order));

                                                  
% missclassification error 
missclasfError = crossval('mcr',X,y,'predfun',classF,'partition',cp);
cfMat = crossval(confusionF,X,y,'partition',cp); % Matrix shows number of correctly and incorrectly classified samples for each classification for each of the 10 cross validated data sets
cfMat = reshape(sum(cfMat),2,2); % summation of the 10 confusion matrices over the 10CV data sets
% Generate confusion matrix

%Note, I don't think this function works in Matlab R2017b, it doesn't seem
%to recognise it
%confusionchart(cfMat, {'Healthy'; 'Heart_Disease'})

%toc

    %}


categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];

%Run Bayesian optimisation

%specify optimisation over the distribution and kernel width
hpOO1 = struct('CVPartition',cp,'Verbose',2,'Optimizer','bayesopt');


CVNBMdl1 = fitcnb(train_features,train_labels,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'PredictorNames',X_header,'CategoricalPredictors',categorical_fields);

%Run grid search for comparison

hpOO2 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

CVNBMdl2 = fitcnb(train_features,train_labels,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'PredictorNames',X_header,'CategoricalPredictors',categorical_fields);

%{
 
%Optimise over distribution type (either kernel or gaussian)
dist = optimizableVariable('dst_name',{'normal','kernel'},'Type','categorical');
cvlossfcn = @(x)kfoldLoss(fitcnb(X,y,'CVPartition',cp,'DistributionNames',char(x.dst_name)),'lossfun','classiferror');
results = bayesopt(cvlossfcn,dist);

%Grid search or Bayesian optimisation across params. Suggested
%optimisations:

%width for kernel distribution
% mn for all variables

%}

