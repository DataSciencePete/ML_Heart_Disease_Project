function Naive_Bayes_optimisation(X,y,cp)

%Run Bayesian optimisation

%specify optimisation over the distribution and kernel width
hpOO1 = struct('CVPartition',cp,'Verbose',2,'Optimizer','bayesopt');

tic;
CVNBMdl1 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
bo_runtime = toc;
fprintf('Bayesopt run time %4.2f\n',bo_runtime);

%Run grid search for comparison

hpOO2 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

tic;
CVNBMdl2 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);
gs_runtime = toc;
fprintf('Gridsearch run time %4.2f\n',gs_runtime);

end

