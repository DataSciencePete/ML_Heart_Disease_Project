function loss = lossFcn_RF_MCR(params,In,Out,opts,cvp)


% A = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(numTrees,XTRAIN,YTRAIN,'method','classification','Options',opts,...
%  'MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS), XTEST));
A = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(params.numTrees,XTRAIN,YTRAIN,'method','classification', 'OOBPrediction','on', 'Options',opts,...
  'MinLeafSize',params.minLS, 'NumPredictorsToSample', params.numPTS), XTEST));

loss = crossval('mcr',In,Out,'predfun',A,'partition',cvp);
end

