[X, y,X_header] = load_heart_csv('heart.csv','numeric','array');

%data partition
cp = cvpartition(y,'KFold',5); % Create CV for data. Each fold has 
%roughly equal size and roughly the same class proportions as in GROUP

%%

%Set up Cross validation
%tic % time how long it takes process to run

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN),XTEST));

order = unique(y); % Order of the group labels
%confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,...
%                                                       cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
%                                                       predict(fitcnb(XTRAIN,YTRAIN),XTEST)),...
%                                                       'order', order));

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

%%


%specify optimisation over the distribution and kernel width
hpOO = struct('CVPartition',cp,'Verbose',2);
%hpOO = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');
%categorical_fields = {'sex','cp','fbs','restecg','exang','slope','ca','thal'};
categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];

%CVNBMdl = fitcnb(X,y,'CVPartition',cp,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO);
%CVNBMdl = fitcnb(X,y,'CVPartition',cp,'OptimizeHyperparameters',{'DistributionNames','Width'});
CVNBMdl = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO,...
    'PredictorNames',X_header,'CategoricalPredictors',categorical_fields);

%'DistributionNames',{'normal','kernel'}



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

