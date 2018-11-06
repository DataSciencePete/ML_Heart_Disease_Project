function Naive_Bayes_optimisation(X,y,cp)
 
categorical_fields = [false,true,true,false,false,true,true,false,true,...
    false,true,true,true];

%Run Bayesian optimisation

%specify optimisation over the distribution and kernel width
hpOO1 = struct('CVPartition',cp,'Verbose',2,'Optimizer','bayesopt');

CVNBMdl1 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);

%Run grid search for comparison

hpOO2 = struct('CVPartition',cp,'Verbose',2,'Optimizer','gridsearch');

CVNBMdl2 = fitcnb(X,y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hpOO1,...
    'CategoricalPredictors',categorical_fields);


end

