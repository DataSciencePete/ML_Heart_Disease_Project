function Naive_Bayes_optimisation(X,y,cp)

%Run Bayesian optimisation
categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];


%specify optimisation over the distribution and kernel width
hpOO1 = struct('CVPartition',cp,'Verbose',2,'Optimizer','bayesopt');

%specify parameters for grid search
hpOO2 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');


%Use Bayesian optimisation to search over possible models with
%missclassification rate (MCR) as the metric
tic;
CVNBMdl1 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
bo_runtime = toc;
fprintf('Bayesopt run using MCR as loss function time %4.2f\n',bo_runtime);

%Now grid search with MCR
tic;
CVNBMdl2 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO2,...
    'CategoricalPredictors',categorical_fields);
gs_runtime = toc;
fprintf('Grid search run using MCR as loss function time %4.2f\n',gs_runtime);

%Bayesian optimisation with cross entropy (CE) as metric
tic;
CVNBMdl3 = fitcnb_ce(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
bo_runtime = toc;
fprintf('Bayesopt run using CE as loss function time %4.2f\n',bo_runtime);

%Grid search with CE
tic;
CVNBMdl4 = fitcnb_ce(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO2,...
    'CategoricalPredictors',categorical_fields);
gs_runtime = toc;
fprintf('Grid search run using CE as loss function time %4.2f\n',gs_runtime);

end

