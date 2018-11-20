function [objFcn, PartitionedModel] = createObjFcn_ce(BOInfo, FitFunctionArgs, Predictors, Response, ...
    ValidationMethod, ValidationVal, Repartition, Verbose)
% Create and return the objective function. If 'Repartition' is false and
% no cvpartition is passed, we first create a cvpartition to be used in all
% function evaluations. The cvp is stored in the workspace of the function
% handle and can be accessed later from the function handle like this:
% f=functions(h);cvp=f.workspace{1}.cvp

%   Copyright 2016-2018 The MathWorks, Inc.

% Set validation value
if ~Repartition && ~isa(ValidationVal, 'cvpartition')
    ValidationVal    = createStaticCVP(BOInfo, Predictors, Response, FitFunctionArgs, ValidationMethod, ValidationVal);
    ValidationMethod = 'CVPartition';
end

% Choose objfcn
if istall(Predictors)
    objFcn = @tallObjFcn;
else
    objFcn = @inMemoryObjFcn;
end

    function Objective = inMemoryObjFcn(XTable)
        % (1) Set up args
        NewFitFunctionArgs = updateArgsFromTable(BOInfo, FitFunctionArgs, XTable);
        % (2) Call fit fcn, suppressing specific warnings
        C = classreg.learning.paramoptim.suppressWarnings();
        PartitionedModel = BOInfo.FitFcn(Predictors, Response, ValidationMethod, ValidationVal, NewFitFunctionArgs{:});
        % (3) Compute kfoldLoss if possible
        if PartitionedModel.KFold == 0
            Objective = NaN;
            if Verbose >= 2
                classreg.learning.paramoptim.printInfo('ZeroFolds');
            end
        else
            if BOInfo.IsRegression
                Objective = log1p(kfoldLoss(PartitionedModel));
            else
%                 Objective = kfoldLoss(PartitionedModel);
                % label = predicted Y ; PartitionedModel.Y = expected Y
                [label, prob] = kfoldPredict(PartitionedModel);
%                 PartitionedModel.Y == label
                kfold_ce = [];
                for row = 1:size(PartitionedModel.Y,1)
%                     PartitionedModel.Y(row)
                    % Retrieve probability of correct classification
                    prob(row,(PartitionedModel.Y(row) + 1));
                    ce = log(prob(row,(PartitionedModel.Y(row) + 1)));
                    % Generate a vector of ce values for each k-fold
                    kfold_ce = [kfold_ce ce];
                end
                % Calculate average ce for each k-fold test set
                Objective = -sum(kfold_ce);
                
            end
            if ~isscalar(Objective)
                % For cases like fitclinear where the user passes Lambda as a vector.
                Objective = Objective(1);
                if Verbose >= 2
                    classreg.learning.paramoptim.printInfo('ObjArray');
                end
            end
        end
    end

    function Objective = tallObjFcn(XTable)
        % (1) Set up fit fcn args
        NewFitFunctionArgs = updateArgsFromTable(BOInfo, FitFunctionArgs, XTable);
        % (2) Set up validation
        if Repartition
            cvp = cvpartition(Predictors(:,1), 'Holdout', ValidationVal, 'Stratify', false);
        else
            cvp = ValidationVal;
        end
        % (3) Split weight arg if present
        [TrainingWeightArgs, TestWeightArgs] = splitWeightArgs(NewFitFunctionArgs, cvp);
        % (4) Call fit fcn on training set, suppressing specific warnings
        C = classreg.learning.paramoptim.suppressWarnings();
        if istall(Response)
            TrainResp = Response(cvp.training);
            TestResp = Response(cvp.test);
        else
            TrainResp = Response;
            TestResp = Response;
        end
        try 
            Model = BOInfo.FitFcn(Predictors(cvp.training,:), TrainResp, NewFitFunctionArgs{:}, TrainingWeightArgs{:});
            % (5) Compute validation loss
            if BOInfo.IsRegression
                Objective = gather(log1p(loss(Model, Predictors(cvp.test,:), TestResp, TestWeightArgs{:})));
            else
                Objective = gather(loss(Model, Predictors(cvp.test,:), TestResp, TestWeightArgs{:}));
            end
            if ~isscalar(Objective)
                % For cases like fitclinear where the user passes Lambda as a vector.
                Objective = Objective(1);
                if Verbose >= 2
                    classreg.learning.paramoptim.printInfo('ObjArray');
                end
            end
        catch msg
            disp(msg.message);
            % Return NaN for MATLAB errors 
            Objective = NaN;
        end
    end
end

function [TrainingWeightArgs, TestWeightArgs] = splitWeightArgs(NVPs, cvp)
% If the 'Weights' NVP is present, split it into training and test, and
% return two cell arrays, each containing a NVP.
[WeightsFound, W] = classreg.learning.paramoptim.parseWeightArg(NVPs);
if ~WeightsFound
    TrainingWeightArgs = {};
    TestWeightArgs = {};
elseif istall(W)
    TrainingWeightArgs = {'Weights',W(cvp.training)};
    TestWeightArgs = {'Weights',W(cvp.test)};
else
    TrainingWeightArgs = {'Weights',W};
    TestWeightArgs = {'Weights',W};
end
end

function cvp = createStaticCVP(BOInfo, Predictors, Response, FitFunctionArgs, ValidationMethod, ValidationVal)
if istall(Predictors)
    assert(isequal(lower(ValidationMethod), 'holdout'));
    cvp = cvpartition(Predictors(:,1), 'Holdout', ValidationVal, 'Stratify', false);
else
    [~,PrunedY] = BOInfo.PrepareDataFcn(Predictors, Response, FitFunctionArgs{:}, 'IgnoreExtraParameters', true);
    if BOInfo.IsRegression
        cvp = cvpartition(numel(PrunedY), ValidationMethod, ValidationVal);
    else
        cvp = cvpartition(PrunedY, ValidationMethod, ValidationVal);
    end
end
end