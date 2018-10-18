[X, y] = load_heart_csv('heart.csv','numeric','array');

%%

%Set up Cross validation
%tic % time how long it takes process to run

%data partition
cp = cvpartition(y,'KFold',5); % Create CV for data. Each fold has 
%roughly equal size and roughly the same class proportions as in GROUP

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN),XTEST));

order = unique(y); % Order of the group labels
%confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,...
%                                                       cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
%                                                       predict(fitcnb(XTRAIN,YTRAIN),XTEST)),...
%                                                       'order', order));

confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,predict(fitcnb(XTRAIN,YTRAIN),XTEST),'order', order));

                                                  
% missclassification error 
%missclasfError = crossval('mcr',X,y,'predfun',classF,'partition',cp);
%cfMat = crossval(confusionF,X,y,'partition',cp); % Matrix shows number of correctly and incorrectly classified samples for each classification for each of the 10 cross validated data sets
%cfMat = reshape(sum(cfMat),2,2); % summation of the 10 confusion matrices over the 10CV data sets
% Generate confusion matrix



%Note, I don't think this function works in Matlab R2017b, it doesn't seem
%to recognise it
%confusionchart(cfMat, {'Healthy'; 'Heart_Disease'})

%toc
%%

%Create set of values for distributions to do gridsearch over
dists_cat = {'mn'};
dists_cont = {'gaussian','kernel'};

%Create an ndimensional grid to store all combinations of distribution
[d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal] = ...
    ndgrid(dists_cont,dists_cat,dists_cat,dists_cont,dists_cont,dists_cat,dists_cat,dists_cont,dists_cat,...
    dists_cont,dists_cat,dists_cat,dists_cat);

%%

%Still need to figure out how to pass each set of arguments to fitncb
%Probably convert dists_cat and dists_cont to an array for ease
%Possibly ndimensionalgrid is not quite the right approach


results = arrayfun(@(varargin) fitcnb(X,y,'CVPartition',cp,'DistributionNames',get_char_args(varargin)),d_age, d_sex, d_cp, d_trestbps, d_chol,...
    d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,'UniformOutput',false);
    

function char_args = get_char_args(varargin)
fprintf('%d',nargin)
celldisp(varargin)
char_cell = {};    
for K = 1: nargin
    char_cell{K} = fprintf('%s',inputname(K));
end
celldisp(char_cell)
char_args = char_cell;
end


%%



%{
 
%Optimise over distribution type (either kernel or gaussian)
dist = optimizableVariable('dst_name',{'normal','kernel'},'Type','categorical');
cvlossfcn = @(x)kfoldLoss(fitcnb(X,y,'CVPartition',cp,'DistributionNames',char(x.dst_name)),'lossfun','classiferror');
results = bayesopt(cvlossfcn,dist);

%Grid search or Bayesian optimisation across params. Suggested
%optimisations:

%need some more understanding of Bayesian optimisation

%width for kernel distribution



% mn for all variables

%}

