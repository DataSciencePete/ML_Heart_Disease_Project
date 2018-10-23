function [features, labels, X_header] = load_heart_csv(filepath,labelType,featureType)
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

%Return feature value in table for random forest
if strcmp(featureType,'table')
    feat = array2table(feat,'VariableNames',X_hdr);
elseif strcmp(featureType,'array')
    feat = feat;
else
    error('Invalid feature type')
end

%return values
features = feat;
labels = lab;
X_header = X_hdr;