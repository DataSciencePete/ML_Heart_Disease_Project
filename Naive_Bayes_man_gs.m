
[train_features, train_labels, test_features, test_labels, X_header] ...
    = load_heart_csv('heart.csv','numeric','array');

cp = cvpartition(train_labels,'KFold',5); % Create CV for data.

%Create set of values for distributions to do gridsearch over
dists_cat = 1;
dists_cont = [2,3];
dists = {'mvmn','normal','kernel'}';

categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];


%Create an ndimensional grid to store all combinations of distribution
[d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal] = ...
    ndgrid(dists_cont,dists_cat,dists_cat,dists_cont,dists_cont,dists_cat,dists_cat,dists_cont,dists_cat,...
    dists_cont,dists_cat,dists_cat,dists_cat);


confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,predict(fitcnb(XTRAIN,YTRAIN),XTEST),'order', order));


%Could change this to use a kfoldFun and pass a function to get the F1
%score for example
results = arrayfun(@(d_age, d_sex, d_cp, d_trestbps, d_chol,d_fbs, d_restecg, ...
    d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal) kfoldLoss(fitcnb(train_features,train_labels,'CVPartition',cp,...
    'CategoricalPredictors',categorical_fields,'DistributionNames',get_char_args(d_age,d_sex,...
    d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak,...
    d_slope, d_ca, d_thal,dists))),d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, ...
    d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,'UniformOutput',false);



%Print out the results of grid search

fileID = fopen('gs_results.csv','w');
fprintf(fileID,strjoin(X_header,','));

for idx = 1:numel(results)
    result_cell = results(idx);
    mdl_score = result_cell{1};
    
    mdl_dists_score = [dists{d_age(idx)}, dists{d_sex(idx)}, dists{d_cp(idx)}, dists{d_trestbps(idx)}, dists{d_chol(idx)}, ...
        dists{d_fbs(idx)}, dists{d_restecg(idx)}, dists{d_thalach(idx)}, dists{d_exang(idx)}, ...
        dists{d_oldpeak(idx)}, dists{d_slope(idx)}, dists{d_ca(idx)}, dists{d_thal(idx)},string(mdl_score)];
    
    fprintf(fileID,strjoin(mdl_dists_score,','));
end

function char_args = get_char_args(d_age, d_sex, d_cp, d_trestbps, d_chol,...
    d_fbs,d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,dists)

    cell_args = {dists{d_age}, dists{d_sex}, dists{d_cp}, dists{d_trestbps}, ...
        dists{d_chol}, dists{d_fbs},dists{d_restecg}, dists{d_thalach}, ...
        dists{d_exang}, dists{d_oldpeak}, dists{d_slope}, dists{d_ca}, dists{d_thal}};
    char_args = cell_args;
end

function cm = confusionFun(CMP,Xtrain,Ytrain,Wtrain,Xtest,Ytest,Wtest)
%Creates a function to compute confusion matrix from CMP
Yhat = predict(CMP,Xtest);
cm = confusionmat(Ytest,Yhat);
end

function averageCost = test(CMP,Xtrain,Ytrain,Wtrain,Xtest,Ytest,Wtest)
%noversicolor Example custom cross-validation function
%   Attributes a cost of 10 for misclassifying versicolor irises, and 1 for
%   the other irises.  This example function requires the |fisheriris| data
%   set.
Ypredict = predict(CMP,Xtest);
misclassified = not(strcmp(Ypredict,Ytest)); % Different result
classifiedAsVersicolor = strcmp(Ypredict,'versicolor'); % Index of bad decisions
cost = sum(misclassified) + ...
    9*sum(misclassified & classifiedAsVersicolor); % Total differences
averageCost = cost/numel(Ytest); % Average error
end