[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','array','numeric');

rng(1);

order = unique(train_labels); % Order of the group labels

%Run final Naive Bayes Model
distributions = {'normal','mvmn','mvmn','kernel','normal','mvmn','mvmn','kernel','mvmn','normal','mvmn','mvmn','mvmn'};

tic;
NB_final_mdl = fitcnb(train_features,train_labels,...
    'DistributionNames',distributions,'Width',36.337);
NB_train_time = toc;

tic;
NB_confusion_mat = confusionmat(test_labels,...
            predict(NB_final_mdl,test_features)...
            ,'Order', order);
NB_predict_test_time = toc;

% Initialise figure plots
figure;
hold on;

[NB_recall_Test, NB_precision_Test, NB_F1_Test, NB_specificity_Test, NB_accuracy_Test, NB_AUC_Test] = get_performance(NB_final_mdl,NB_confusion_mat, test_features, test_labels);

fprintf('Naive Bayes train time %4.2fs\n',NB_train_time);

[train_features, train_labels, test_features, test_labels, X_header, cp] = load_heart_csv('heart.csv','table','categorical');

order = unique(train_labels); % Order of the group labels(for categorical column)

%Run final Random Forest Model
In_high_imp_variables = removevars(train_features,{'age','trestbps','chol','fbs', 'restecg','exang','slope'});

tic;
RF_final_mdl = TreeBagger(44,In_high_imp_variables, train_labels,...
                        'method','classification',...
                        'OOBPrediction','on',...
                        'MinLeafSize',6,...
                        'NumPredictorsToSample', 1);
RF_train_time = toc; % Time taken to train final RF model on all the training data

fprintf('Random Forest train time %4.2fs\n',RF_train_time);

RF_confusion_mat = confusionmat(test_labels,...
            categorical(...
            cellfun(@str2num,... % convert cell array of character vectors to a cell array of numerics
            predict(RF_final_mdl,test_features))),...
            'order', order);
 
RF_predict_test_time = toc; % Time taken for optimised model to make test set predictions
        
% Get performance data for test data
[RF_recall_Test, RF_precision_Test, RF_F1_Test, RF_specificity_Test, RF_accuracy_Test, RF_AUC_Test] = get_performance(RF_final_mdl,RF_confusion_mat, test_features, test_labels);
ax = gca; % grab current axis
ax.FontSize = 16 % Alter font size
ax.FontWeight = 'bold';
lg = legend('Naive Bayes', 'Random Forest');
  
hold off;


model_metrics = [NB_recall_Test, NB_precision_Test, NB_F1_Test, NB_specificity_Test, NB_accuracy_Test, NB_AUC_Test;...
    RF_recall_Test, RF_precision_Test, RF_F1_Test, RF_specificity_Test, RF_accuracy_Test, RF_AUC_Test];    

% Draw bar chart comparing performance metrics on test data
figure;
bar_chart = barh(model_metrics');
xlim([0.6,1]) % set y axis limits
ax = gca; % grab handle to current axis
% Add labels to each x tick
ax.YTickLabel = {'Recall', 'Precision', 'F1', 'Specificity', 'Accuracy', 'AUC'};
legendvals = {'RF'; 'NB'}; % Set legend names
% change order of legend entries
lg = legend([bar_chart(2), bar_chart(1)], legendvals, 'Location', 'southeast');
ax.FontSize = 16 % Alter font size
ax.FontWeight = 'bold';


