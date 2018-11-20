function loss = myCVlossfcn_grid(l,p,In,Out,opts)

%data partition
cp = cvpartition(Out,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP 


A = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(50,XTRAIN,YTRAIN,'method','classification','Options',opts,...
 'MinLeafSize',l,'NumPredictorstoSample',p), XTEST));

loss = crossval('mcr',In,Out,'predfun',A,'partition',cp);
end