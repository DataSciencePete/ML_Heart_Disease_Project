function loss = myCVlossfcn(params,In,Out,opts)

%data partition
cp = cvpartition(Out,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 


A = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(numTrees,XTRAIN,YTRAIN,'method','classification','Options',opts,...
 'MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS), XTEST));

loss = crossval('mcr',In,Out,'predfun',A,'partition',cp);
end

