
function Naive_Bayes_man_gs(X,y,X_header,cp)

%Create set of values for distributions to do gridsearch over
dists_cat = 1;
dists_cont = [2,3];
dists = {'mvmn','normal','kernel'}';

categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];

%Create an ndimensional grid to store all combinations of distribution
[d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal] = ndgrid(dists_cont,dists_cat,dists_cat,dists_cont,dists_cont,dists_cat,dists_cat,dists_cont,dists_cat,...
    dists_cont,dists_cat,dists_cat,dists_cat);

%Flatten each of the n dimensional feature vectors for reporting results
d_age=vflat(d_age);
d_sex=vflat(d_sex);
d_cp=vflat(d_cp);
d_trestbps=vflat(d_trestbps);
d_chol=vflat(d_chol);
d_fbs=vflat(d_fbs);
d_restecg=vflat(d_restecg);
d_thalach=vflat(d_thalach);
d_exang=vflat(d_exang);
d_oldpeak=vflat(d_oldpeak);
d_slope=vflat(d_slope);
d_ca=vflat(d_ca);
d_thal=vflat(d_thal);

feature_array = [d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal];
%nd_grid = [d_age; d_sex; d_cp; d_trestbps; d_chol; d_fbs; d_restecg; d_thalach; d_exang; d_oldpeak; d_slope; d_ca; d_thal];

tic;
results = arrayfun(@(d_age, d_sex, d_cp, d_trestbps, d_chol,d_fbs, d_restecg, ...
    d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal) kfoldLoss(fitcnb(X,y,'CVPartition',cp,...
    'CategoricalPredictors',categorical_fields,'DistributionNames',get_char_args(d_age,d_sex,...
    d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak,...
    d_slope, d_ca, d_thal,dists))),d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, ...
    d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,'UniformOutput',false);
man_gs_runtime = toc;
fprintf('Manual grid search run time %4.2f\n',man_gs_runtime);


tic;
results_CE = arrayfun(@(d_age, d_sex, d_cp, d_trestbps, d_chol,d_fbs, d_restecg, ...
    d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal)CE_loss(fitcnb(X,y,'CVPartition',cp,...
    'CategoricalPredictors',categorical_fields,'DistributionNames',get_char_args(d_age,d_sex,...
    d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak,...
    d_slope, d_ca, d_thal,dists))),d_age, d_sex, d_cp, d_trestbps, d_chol, d_fbs, ...
    d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,'UniformOutput',false);
man_gs_runtime = toc;
fprintf('Manual grid search run time %4.2f\n',man_gs_runtime);

print_gs_results(results,X_header,dists,feature_array);
print_gs_results(results_CE,X_header,dists,feature_array);

end

function char_args = get_char_args(d_age, d_sex, d_cp, d_trestbps, d_chol,...
    d_fbs,d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal,dists)

    cell_args = {dists{d_age}, dists{d_sex}, dists{d_cp}, dists{d_trestbps}, ...
        dists{d_chol}, dists{d_fbs},dists{d_restecg}, dists{d_thalach}, ...
        dists{d_exang}, dists{d_oldpeak}, dists{d_slope}, dists{d_ca}, dists{d_thal}};
    char_args = cell_args;
end

%Print out the results of grid search
function print_gs_results(results,X_header,dists,feature_array)
fileID = 1;
fprintf(fileID,'%s\n',strjoin(X_header,','));

mdl_dists = arrayfun(@(x) dists{x},feature_array,'UniformOutput',false);

%num_features = numel(mdl_dists)/numel(results);
%num_results = numel(results);

for idx = 1:numel(results)
    result_cell = results(idx);
    mdl_score = result_cell{1};
    
    mdl_dists_score = [mdl_dists(idx,:),string(mdl_score)];
    
    fprintf(fileID,'%s\n',strjoin(mdl_dists_score,','));
    
end

end


function loss = CE_loss(PartitionedModel)

    [label, prob] = kfoldPredict(PartitionedModel);
    %                 PartitionedModel.Y == label
                    kfold_ce = [];
                    for row = 1:size(PartitionedModel.Y,1)
    %                     PartitionedModel.Y(row)
                        % Retrieve probability of correct classification
                        prob(row,(PartitionedModel.Y(row) + 1));
                        ce = log(prob(row,(PartitionedModel.Y(row) + 1)));
                        % Generate a vector of ce values for each k-fold
                        kfold_ce = [kfold_ce ce];
                    end
                    % Calculate average ce for each k-fold test set
                    loss = -sum(kfold_ce);
end

function reshaped_ndvector = vflat(feature_vector)
    reshaped_ndvector = reshape(feature_vector,numel(feature_vector),1);
end
