function [inputs, labels, headers] = load_heart_csv(filepath,labelType)
% Returns training and test data

data = csvread(filepath,1,0);
lab = data(:,size(data,2));
inp = data(:,1:size(data,2)-1);
clear data

%Get column headers
fileID = fopen('heart.csv','r','n','UTF-8');
hdr = strsplit(fgetl(fileID),',');

%convert train_labels into binary encoded array if required
if strcmp(labelType,'onehot')
    lab = full(ind2vec((lab+1)'))';
elseif strcmp(labelType,'numeric')
    lab = lab;
else
    error('Invalid label return type')
end
       
%return values
inputs = inp;
labels = lab;
headers = hdr;