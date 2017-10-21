function [ err ] = baselines( Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
% Inputs
% Xtrain: input training data
% Ytrain: output training data
% Xtest: input test data
% Ytest: output test data
% lambda: regularization parameter (for global and local models)
% opts:
%   opts.avg: whether to compute avg or total error
%   opts.obj: 'R' for regression, 'C' for classification
%   opts.type:
%       consant - for each task, the model is the average (R) or mode (C) of the labels
%       global - concatenate data from all tasks, fit one global model
%       local - fit m separate models

% Output
% Average or total RMSE (R) or classification error (C) across tasks

%% set variables
m = length(Xtrain); % # of tasks
d = size(Xtrain{1}, 2); % # of features

%% compute baselines
switch opts.type
    case 'constant'
        % predict the mean output (R) or mode class (C) from training
        if(opts.obj == 'R') % regression
            if(opts.avg)
                errs = zeros(m, 1);
                for t=1:m
                    pred = mean(Ytrain{t});
                    errs(t) = sqrt(mean((Ytest{t} - pred).^2));
                end
                err = mean(errs);
            else
                Y_hat = cell(m,1);
                for t=1:m
                    Y_hat{t} = repmat(mean(Ytrain{t}), length(Ytest{t}), 1);
                end
                Y = cell2mat(Ytest(:));
                Y_hat = cell2mat(Y_hat);
                err = sqrt(mean((Y - Y_hat).^2));
            end
        else % classification
            if(opts.avg)
                errs = zeros(m, 1);
                for t=1:m
                    pred = mode(Ytrain{t});
                    errs(t) = mean(pred ~= Ytest{t});
                end
                err = mean(errs);
            else
                pred = mode(opts.Ytrainactual);
                err = sum(pred ~= opts.Yactual) / length(opts.Yactual);
            end
        end
        
    case 'global'
        % compute one global model for the data
        allX = cat(1, Xtrain{:});
        allY = cat(1, Ytrain{:});
        allXtest = cat(1, Xtest{:});
        allYtest = cat(1, Ytest{:});
        if(opts.obj == 'R') % regression
            w = inv(allX' * allX + lambda * eye(d)) * allX' * allY;
            err = sqrt(mean((allYtest - allXtest * w).^2));
        else % classification
            w = simple_svm(allX, allY, lambda, opts);
            if(opts.avg)
                errs = zeros(m, 1);
                for t=1:m
                    predvals = sign(Xtest{t} * w);
                    errs(t) = mean(predvals ~= Ytest{t});
                end
                err = mean(errs);
            else
                predvals = sign(allXtest * w);
                err = mean(allYtest ~= predvals);
            end
        end
        
    case 'local'
        % compute m completely separate local models
        if(opts.obj == 'R') % regression
            if(opts.avg)
                errs = zeros(m,1);
                for t=1:m
                    wt = inv(Xtrain{t}' * Xtrain{t} + lambda * eye(d)) * Xtrain{t}' * Ytrain{t};
                    errs(t) = sqrt(mean((Ytest{t} - Xtest{t} * wt).^2));
                end
                % compute average rmse
                err = mean(errs);
            else
                Y_hat = cell(m,1);
                for t=1:m
                    wt = inv(Xtrain{t}' * Xtrain{t} + lambda * eye(d)) * Xtrain{t}' * Ytrain{t};
                    Y_hat{t} = Xtest{t} * wt;
                end
                % compute total rmse
                Y = cell2mat(Ytest(:));
                Y_hat = cell2mat(Y_hat);
                size(Y_hat)
                size(Y)
                err = sqrt(mean((Y - Y_hat).^2));
            end
        else % classification
            Y_hat = cell(m,1);
            for t=1:m
                wt = simple_svm(Xtrain{t}, Ytrain{t}, lambda, opts);
                Y_hat{t} = sign(Xtest{t} * wt);
            end
            if(opts.avg)
                errs = zeros(m,1);
                for t=1:m
                    errs(t) = mean(Ytest{t} ~= Y_hat{t});
                end
                err = mean(errs);
            else
                Y = cell2mat(Ytest);
                Y_hat = cell2mat(Y_hat);
                err = mean(Y ~= Y_hat);
            end
        end
end

end