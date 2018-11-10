
function Naive_Bayes_man_gs(X,y,X_header,cp)

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


%Could change this to use a kfoldFun and pass a function to get the F1
%score for example, but kfoldFun seems to raise an issue
tic;
results = arrayfun(@(d_age, d_sex, d_cp, d_trestbps, d_chol,d_fbs, d_restecg, ...
    d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal) kfoldLoss(fitcnb(X,y,'CVPartition',cp,...
    'CategoricalPredictors',categorical_fields,'DistributionNames',get_char_args(d_age,d_sex,...
    d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak,...
    d_slope, d_ca, d_thal,dists))),d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, ...
    d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,'UniformOutput',false);
man_gs_runtime = toc;
fprintf('Manual grid search run time %4.2f\n',man_gs_runtime);

%Print out the results of grid search

fileID = 1;
fprintf(fileID,'%s\n',strjoin(X_header,','));

for idx = 1:numel(results)
    result_cell = results(idx);
    mdl_score = result_cell{1};
    
    mdl_dists_score = [dists{d_age(idx)}, dists{d_sex(idx)}, dists{d_cp(idx)}, dists{d_trestbps(idx)}, dists{d_chol(idx)}, ...
        dists{d_fbs(idx)}, dists{d_restecg(idx)}, dists{d_thalach(idx)}, dists{d_exang(idx)}, ...
        dists{d_oldpeak(idx)}, dists{d_slope(idx)}, dists{d_ca(idx)}, dists{d_thal(idx)},string(mdl_score)];
    
    fprintf(fileID,'%s\n',strjoin(mdl_dists_score,','));
end

function char_args = get_char_args(d_age, d_sex, d_cp, d_trestbps, d_chol,...
    d_fbs,d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,dists)

    cell_args = {dists{d_age}, dists{d_sex}, dists{d_cp}, dists{d_trestbps}, ...
        dists{d_chol}, dists{d_fbs},dists{d_restecg}, dists{d_thalach}, ...
        dists{d_exang}, dists{d_oldpeak}, dists{d_slope}, dists{d_ca}, dists{d_thal}};
    char_args = cell_args;
end

end
