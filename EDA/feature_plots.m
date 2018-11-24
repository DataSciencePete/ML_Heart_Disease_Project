%% Load data into Matlab
% Add folders to the path
addpath(genpath('../'));
[train_features, train_labels, test_features, test_labels, X_header,cvp, heart_matrix] = load_heart_csv('heart.csv', 'table', 'categorical');


% Select columns from heart_matrix relating to categorical data
z = [1 4 5 8 10];
titles = {'age - Age'; 'trestbp - Resting Blood Pressure'; 'chol - Serum Cholesterol mg/dl';...
    'thalach - Maximum Heart Rate Achieved'; 'oldpeak - ST Depression Induced by Exercise Relative to Rest'}; 
% Draw boxplots for all categorical data
for i = z
 figure(find(z == i) + 8);
 boxplot(heart_matrix(:,i), heart_matrix(:,end))
 xlab = {'Healthy', 'Heart Disease'};
 set(gca,'XtickLabel',xlab);
 a = get(get(gca,'children'),'children');
 set(a(5), 'Color', 'r', 'LineWidth', 2); 
 set(a(6), 'Color', 'b', 'LineWidth', 2); 
 title(titles(find(z == i)), 'FontSize', 18);
 
end

%% Draw bar charts for continuous data
%% Sex - healthy/heart disease stacked bar chart
% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,2);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,2);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:2);
heart_disease_counts = histcounts(heart_disease,0:2);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Plot bar chart
figure(1);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'Female'; 'Male'};
set(gca,'XtickLabel',xvals,'FontSize',18);
legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('sex - Female or Male', 'FontSize', 18);

%% cp Chest pain - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,3);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,3);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:4);
heart_disease_counts = histcounts(heart_disease,0:4);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% reverse order of matrix for ease of visualisation
counts([1 2 3 4]) = counts([4 3 2 1]);
counts([2 3],:) = counts([3 2],:);
% Plot bar chart
figure(2);
bc =bar(counts, 'stacked');
% Add x axis labels
xvals = {'asymptomatic'; 'non-anginal pain';'atypical angina '; 'angina'};
set(gca,'XtickLabel',xvals,'FontSize',18,'FontWeight','bold');
legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('cp - Chest pain type', 'FontSize', 30);
xtickangle(45);

%% fbs - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,6);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,6);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:2);
heart_disease_counts = histcounts(heart_disease,0:2);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Alter order for ease of visualisation
counts([1 2],:) = counts([2 1],:);


% Plot bar chart
figure(3);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'fbs <= 120 mg/dl)'; 'fbs > 120 mg/dl)'};
set(gca,'XtickLabel',xvals,'FontSize',18);
legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('fbs - Fasting blood sugar', 'FontSize', 18);



%% restecg - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,7);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,7);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:3);
heart_disease_counts = histcounts(heart_disease,0:3);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Plot bar chart
figure(4);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'Normal'; 'ST-T wave abnormality '; 'Probable/Definite''\newline''ventricular hypertrophy'};
set(gca,'XtickLabel',xvals,'FontSize',18);

legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('restecg - Resting electrocardiography results', 'FontSize', 18);

%% exang - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,9);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,9);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:2);
heart_disease_counts = histcounts(heart_disease,0:2);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Plot bar chart
figure(5);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'Absence'; 'Presence'};
set(gca,'XtickLabel',xvals,'FontSize',18);

xlabel('Exercise Induced Angina');
legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('exang - Exercise induced angina', 'FontSize', 18);


%% slope - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,11);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,11);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:3);
heart_disease_counts = histcounts(heart_disease,0:3);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Plot bar chart
figure(6);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'Upsloping'; 'Flat'; 'Downsloping'};
set(gca,'XtickLabel',xvals,'FontSize',18);

legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('slope - Slope of peak exercise ST segment', 'FontSize', 18);

%% ca - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,12);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,12);

% Generate frequency counts
healthy_counts = histcounts(healthy,0:4);
heart_disease_counts = histcounts(heart_disease,0:4);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Plot bar chart
figure(7);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'0'; '1'; '2'; '3'};
set(gca,'XtickLabel',xvals,'FontSize',18);

xlabel('# of blocked vessels');
legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('ca - Number of major vessels colored by fluoroscopy', 'FontSize', 18);

%% thal - healthy/heart disease stacked bar chart

% Slice matrix for healthy and heart disease and extract sex columns
healthy = heart_matrix(heart_matrix(:,end) == 0,13);
heart_disease = heart_matrix(heart_matrix(:,end) == 1,13);

% Generate frequency counts
healthy_counts = histcounts(healthy,1:4);
heart_disease_counts = histcounts(heart_disease,1:4);
% transpose matrix in order to stack frequency values for healthy versus
% heart disease bars
counts = transpose([healthy_counts ; heart_disease_counts]);
% Alter order for ease of visualisation
counts([2 3],:) = counts([3 2],:);

% Plot bar chart
figure(8);
bc = bar(counts, 'stacked');
% Add x axis labels
xvals = {'Normal'; 'Reversible defect'; 'Fixed defect'};
set(gca,'XtickLabel',xvals,'FontSize',18);

legendvals = {'Healthy'; 'Heart Disease'};
legend(bc, legendvals, 'Location', 'northwest','FontSize',18);
title('thal - Thalium stress test result', 'FontSize', 18);

%% 
