%Function to report model performance
function [recall, precision, F1, specificity,accuracy, AUC] = get_performance(mdl,confusion_mat,features, labels)

recall = confusion_mat(1)/(confusion_mat(1)+ confusion_mat(3));
precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
F1 = (2*(precision * recall))/(precision + recall);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);

fprintf('Recall %6.4f\n',recall);
fprintf('Precision %6.4f\n',precision);
fprintf('F1 %6.4f\n',F1);
fprintf('Specificity %6.4f\n',specificity);
fprintf('Accuracy %6.4f\n',accuracy);

% Draw ROC curve
[yhat,scores,cost] = predict(mdl,features);

%need to find NB method to calculate class scores
%should be able to use predict function

% calc fpr and tpr at different threshold as defined by T for ROC curve
[fpr,tpr, T, AUC] = perfcurve(labels,scores(:,2), 1);

% Plot ROC curve

plot(fpr,tpr, 'LineWidth',2);

xlabel('False Positive Rate');
ylabel('True Positive Rate');


end