[X, y] = load_heart_csv('heart.csv','numeric','array');

%Set up Cross validation
%tic % time how long it takes process to run

%data partition
cp = cvpartition(y,'KFold',10); % Create 10-folds cross-validation partition for data. Each subsample has roughly equal size and roughly the same class proportions as in GROUP

%prediction function to be supplied to crossval function
classF = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN),XTEST));

order = unique(y); % Order of the group labels
%confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,...
%                                                       cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
%                                                       predict(fitcnb(XTRAIN,YTRAIN),XTEST)),...
%                                                       'order', order));

confusionF = @(XTRAIN,YTRAIN,XTEST,YTEST)(confusionmat(YTEST,predict(fitcnb(XTRAIN,YTRAIN),XTEST),'order', order));

%%                                                   
% missclassification error 
missclasfError = crossval('mcr',X,y,'predfun',classF,'partition',cp);
cfMat = crossval(confusionF,X,y,'partition',cp); % Matrix shows number of correctly and incorrectly classified samples for each classification for each of the 10 cross validated data sets
cfMat = reshape(sum(cfMat),2,2); % summation of the 10 confusion matrices over the 10CV data sets
% Generate confusion matrix

%%

%Note, I don't think this function works in Matlab R2017b, it doesn't seem
%to recognise it
%confusionchart(cfMat, {'Healthy'; 'Heart_Disease'})

%toc



