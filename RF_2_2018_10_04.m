[In, Infemale, Inmale, Out, Outfemale, Outmale, heart_matrix] = loadheart('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv'); % see script file loadhear.m

par = devicespec(); % see script file devicespec.m


%% Section 1
% Set up Cross validation using Random Forest see https://uk.mathworks.com/help/stats/treebagger.html for hyperparameter settings and https://uk.mathworks.com/matlabcentral/answers/34771-classifier for code
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
                                                                          'Options',par...
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


%% Section 2
% Bayesian Optimisation

% Min No of Observations per leaf
maxMinLS = 50;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
% No of variables to consider at each split in the tree
numPTS = optimizableVariable('numPTS',[1,size(In_high_imp_variables,2)],'Type','integer'); % define number of predictors
hyperparametersRF = [minLS; numPTS];

% Also also consider tuning the number of trees in the ensemble




results = bayesopt(@(params)Optimisation(params,In_high_imp_variables,Out),hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);



%% Section 3 Bayesian Opimisation with Cross validation

results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,Out,par),hyperparametersRF, 'AcquisitionFunctionName', 'probability-of-improvement', 'IsObjectiveDeterministic', true, 'MaxObjectiveEvaluations', 40);

results = bayesopt(@(params)myCVlossfcn(params,In_high_imp_variables,Out,par),hyperparametersRF, 'MaxObjectiveEvaluations', 300);


%% Section 4 Grid Search Random Forest approach

minLS_grid = linspace(1,20,20);  % Min No of observations per leaf (paramter search space)
numPTS_grid = linspace(1,size(In,2),size(In,2)); % Number of variables to select at random for each decision split (paramter search space)

[LS,P] = ndgrid(minLS_grid, numPTS_grid); % Parameter grid

fitresult = arrayfun(@(p1,p2) fittingfunction(p1,p2), F, S); %run a fitting on every pair fittingfunction(F(J,K), S(J,K))
result_grid = arrayfun(@(l,p)myCVlossfcn_grid(l,p,In,Out,par), LS, P);
%%
% testmodel = TreeBagger(50,In,Out,...
%           'Method','classification',...
%           'OOBVarImp','On',...
%           'Options',par,...
%           'OOBPredictorImportance','on'...
%           )
% checkpredict = predict(testmodel, In)
% 
% checkpredict = cellfun(@str2num, checkpredict);
% confusionmat(Out, checkpredict)