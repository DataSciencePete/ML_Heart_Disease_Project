[In, Out] = loadheart('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv'); % see script file loadhear.m

par = devicespec(); % see script file devicespec.m


%% Set up Cross validation using Random Forest see https://uk.mathworks.com/help/stats/treebagger.html for hyperparameter settings and https://uk.mathworks.com/matlabcentral/answers/34771-classifier for code
tic % time how long it takes process to run

%data partition
cp = cvpartition(Out,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(50,XTRAIN,YTRAIN,...
                                                        'Method','classification',...
                                                        'OOBVarImp','On',...
                                                        'Options',par...
                                                       ),XTEST));
order = unique(Out); % Order of the group labels
confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,...
                                                       cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
                                                       predict(TreeBagger(50,XTRAIN,YTRAIN,...
                                                                          'Method','classification',...
                                                                          'OOBVarImp','On',...
                                                                          'Options',paroptions...
                                                                         ),...
                                                                XTEST...
                                                               )),...
                                                       'order', order...
                                                      )...
                                          );
% missclassification error 
missclasfError = crossval('mcr',In,Out,'predfun',classF,'partition',cp);
cfMat = crossval(confusionF,In,Out,'partition',cp); % Matrix shows number of correctly and incorrectly classified samples for each classification for each of the 10 cross validated data sets
cfMat = reshape(sum(cfMat),2,2); % summation of the 10 confusion matrices over the 10CV data sets
% Generate confusion matrix
confusionchart(cfMat, {'Healthy'; 'Heart_Disease'})
toc


%%
% Bayesian Optimisation

maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(In,2)],'Type','integer'); % define number of predictors
hyperparametersRF = [minLS; numPTS];

% Also also consider tuning the number of trees in the ensemble


function oobErr = oobErrRF(params,X)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(300,X,'MPG','Method','regression',...
    'OOBPrediction','on','MinLeafSize',params.minLS,...
    'NumPredictorstoSample',params.numPTS);
oobErr = oobQuantileError(randomForest);
end

results = bayesopt(@(params)oobErrRF(params,X),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);

%%
% testmodel = TreeBagger(50,In,Out,...
%           'Method','classification',...
%           'OOBVarImp','On',...
%           'Options',paroptions...
%           )
% checkpredict = predict(testmodel, In)
% 
% checkpredict = cellfun(@str2num, checkpredict);
% confusionmat(Out, checkpredict)