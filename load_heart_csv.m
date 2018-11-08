function [train_features, train_labels, test_features, test_labels, ...
    X_header, cvp] = load_heart_csv(filepath,featureType,labelType)
% Returns training and test data

%Fix the random seed to ensure the same data and cvpartition is returned
rng(1)

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

%Randomly permute the data to remove any effect of blocks
p = randperm(size(feat,1));
[trainInd, testInd] = divideblock(size(feat,1),0.8,0.2);
train_features = feat(p(trainInd),:);
test_features = feat(p(testInd),:);
train_labels = lab(p(trainInd));
test_labels = lab(p(testInd));

if strcmp(featureType,'table')
    train_features = array2table(train_features,'VariableNames',X_hdr);
    test_features  = array2table(test_features ,'VariableNames',X_hdr);
elseif strcmp(featureType,'array')
    %do nothing
else
    error('Invalid feature type')
end

if strcmp(labelType,'categorical')
    train_labels = categorical(train_labels);
    test_labels = categorical(test_labels);
elseif strcmp(labelType,'numeric')
    %do nothing    
else
    error('Invalid feature type')
end




%return values
cvp = cvpartition(train_labels,'KFold',10); % Create CV for data. Each fold has 
%roughly equal size and roughly the same class proportions as in GROUP
X_header = X_hdr;