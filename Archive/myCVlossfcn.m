function loss = myCVlossfcn_nb(params,X,y,cvp)

predFcn_width = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN,'DistributionNames',char(params.dist),'Width',params.widthparam), XTEST));

predFcn_NoWidth = @(XTRAIN,YTRAIN,XTEST)(predict(fitcnb(XTRAIN,YTRAIN,'DistributionNames',char(params.dist)), XTEST));

if strcmp(char(params.dist), 'kernel')
    loss = crossval('mcr',X,y,'predfun',predFcn_width,'partition',cvp);
else
    loss = crossval('mcr',X,y,'predfun',predFcn_NoWidth,'partition',cvp);
end


