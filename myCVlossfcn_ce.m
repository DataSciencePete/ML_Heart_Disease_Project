function loss = myCVlossfcn_ce(params,In,Out,opts,cvp)
                               
    function cross_ent = ce_funct(XTRAIN,YTRAIN,XTEST,YTEST)
            [~, prob] = predict(TreeBagger(...
                                params.numTrees,XTRAIN,YTRAIN,...
                                'method','classification',...
                                'OOBPrediction','on',...
                                'Options',opts,...
                                'MinLeafSize',params.minLS,...
                                'NumPredictorsToSample', 1),...
                                    XTEST...
                                   ); 
         % Calculate CE for each k-fold partition
         kfold_ce = [];
         for row = 1:size(YTEST,1) 
             size(YTEST,1)
             YTEST(row)
             prob(:,(double(YTEST(row))))
             prob(row,(double(YTEST(row))))
             ce = log(prob(row,(double(YTEST(row)))));
             % Generate a vector of ce values for each k-fold
             kfold_ce = [kfold_ce ce];
%              cross_ent =  size(prob(1,:,:));
         end 
         % Calculate average ce for each k-fold test set
         cross_ent = -sum(kfold_ce);
             
    end

% Acquire the matrix of probabilities for each k-fold test
av_k_fold_ce = crossval(@ce_funct,In,Out,'partition',cvp);
loss = mean(av_k_fold_ce);
end



