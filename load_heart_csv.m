function [train_features, train_labels, test_features, test_labels, ...
    X_header] = load_heart_csv(filepath,labelType,featureType, rng_seed)

%set rng seed number to ensure predictive split is obtained
%This ensures that both algorithms will use the same train test data 
% call function like [train_features, train_labels, test_features, test_labels, X_header] = load_heart_csv('/Users/kevinryan/Documents/DataScienceMSc/MachineLearning/Coursework/heart.csv', 'numeric', 'table', 22); % see script file loadhear.m
rng(rng_seed);
% Returns training and test data

data = csvread(filepath,1,0);
lab = data(:,size(data,2));
feat = data(:,1:size(data,2)-1);
clear data

%Get column headers
fileID = fopen('heart.csv','r','n','UTF-8');
hdr = strsplit(fgetl(fileID),',');
fclose(fileID);

%Remove any spaces from headers
hdr = regexprep(hdr,'\W','');
X_hdr = hdr(1:size(hdr,2)-1);

%convert train_labels into binary encoded array if required
if strcmp(labelType,'onehot')
    lab = full(ind2vec((lab+1)'))';
elseif strcmp(labelType,'numeric')
    lab = lab;
else
    error('Invalid label return type')
end

%Randomly permute the data to remove any effect of blocks
p = randperm(size(feat,1));
[trainInd, testInd] = divideblock(size(feat,1),0.8,0.2);
train_features = feat(p(trainInd),:);
test_features = feat(p(testInd),:);
train_labels = lab(p(trainInd));
test_labels = lab(p(trainInd));

%Return feature value in table form
if strcmp(featureType,'table')
    train_features = array2table(train_features,'VariableNames',X_hdr);
    test_features  = array2table(test_features ,'VariableNames',X_hdr);
elseif strcmp(featureType,'array')
    %do nothing
else
    error('Invalid feature type')
end

%return values
X_header = X_hdr;