[X, y] = load_heart_csv('heart.csv','numeric');

mdl = fitcnb(X,y);

%% Cross validation
rng(1); % fix seed to 1
CVmdl = crossval(mdl);
defaultLoss = kfoldLoss(CVmdl);

%% Confusion matrix

isLabels1 = resubPredict(Mdl1); %what is a resubstitution prediction?
ConfusionMat1 = confusionmat(Y,isLabels1)

