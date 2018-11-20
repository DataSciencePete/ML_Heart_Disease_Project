function loss = myCVlossfcn_nbce(params,In,Out,cvp)

    function cross_ent = ce_funct(XTRAIN,YTRAIN,XTEST,YTEST)
        %Specify categorical features - these features by default will then be
        %assigned as having a multivariate multinomial distribution
        categorical_fields = [false,true,true,false,false,true,true,false,true,...
            false,true,true,true];
        
        if strcmp(char(params.dist), 'kernel')
            [label, posterior, cost] = predict(...
                                        fitcnb(XTRAIN,YTRAIN,...
                                        'DistributionNames', {char(params.dist), 'mvmn', 'mvmn', char(params.dist), char(params.dist), 'mvmn',...
                                        'mvmn', char(params.dist), 'mvmn', char(params.dist), 'mvmn', 'mvmn', 'mvmn'}, ...
                                        'Width', [params.widthparam],'CategoricalPredictors',categorical_fields),...

                                        XTEST...
                                        );

        else
            [label, posterior, cost] = predict(...
                                        fitcnb(XTRAIN,YTRAIN,...
                                        'DistributionNames', {char(params.dist), 'mvmn', 'mvmn', char(params.dist), char(params.dist), 'mvmn',...
                                        'mvmn', char(params.dist), 'mvmn', char(params.dist), 'mvmn', 'mvmn', 'mvmn'},...
                                        'CategoricalPredictors',categorical_fields),...
                                        XTEST...
                                        );
        
        end 
                                    
                                    
         % Calculate CE for each k-fold partition
         kfold_ce = [];
         for row = 1:size(YTEST,1) 
             %size(YTEST,1)
             %YTEST(row)
             %posterior(:,(YTEST(row) + 1))
             %posterior(row,(YTEST(row) + 1))
             ce = log(posterior(row,(YTEST(row) + 1)));
             % Generate a vector of ce values for each k-fold
             kfold_ce = [kfold_ce ce];
         end 
         % Calculate average ce for each k-fold test set
         cross_ent = -sum(kfold_ce);
             
    end

% Acquire the matrix of probabilities for each k-fold test
av_k_fold_ce = crossval(@ce_funct,In,Out,'partition',cvp);
loss = mean(av_k_fold_ce);
end