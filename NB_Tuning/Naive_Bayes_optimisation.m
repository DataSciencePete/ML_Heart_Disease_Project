function Naive_Bayes_optimisation(X,y,cp)

%Run Bayesian optimisation

%specify optimisation over the distribution and kernel width
hpOO1 = struct('CVPartition',cp,'Verbose',2,'Optimizer','bayesopt');
% specify which predictor paramters to alter in NB model
OptimizeHyperparameters = {};

tic;
CVNBMdl1 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
bo_runtime = toc;
fprintf('Bayesopt run using mcr as loss function time %4.2f\n',bo_runtime);

tic;
CVNBMdl1b = fitcnb_ce(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
bo_runtime = toc;
fprintf('Bayesopt run using cross entropy as loss function time %4.2f\n',bo_runtime);


%Run grid search for comparison

hpOO2 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

tic;
CVNBMdl2 = fitcnb_ce(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
gs_runtime = toc;
fprintf('Gridsearch run time %4.2f\n',gs_runtime);

end

