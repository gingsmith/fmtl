function [best_lambda] = cross_val_1( X, Y, method_str, method_opts, lambda_range, cv_fold)
% Model selection (cross validation) for 1 parameter 
%
% INPUT
%   X:             input data
%   Y:             output data
%   method_str:    the method to run the model training 
%   method_opts:   the inputs to the model training method
%   lambda_range:  the possible parameter values
%   cv_fold:       # of cross validation fold
%
% OUTPUT
%  best_lambda

method = str2func(method_str);

% compute sample size for each task
task_num = length(X);

% begin cross validation
fprintf('[')
for cv_idx = 1:cv_fold
    fprintf('.')
    
    % buid cross validation data splittings for each task.
    cv_Xtr = cell(task_num, 1);
    cv_Ytr = cell(task_num, 1);
    cv_Xte = cell(task_num, 1);
    cv_Yte = cell(task_num, 1);
    
    for t = 1: task_num
        task_sample_size = length(Y{t});
        te_idx = cv_idx : cv_fold : task_sample_size;
        tr_idx = setdiff(1:task_sample_size, te_idx);
        
        cv_Xtr{t} = X{t}(tr_idx, :);
        cv_Ytr{t} = Y{t}(tr_idx, :);
        cv_Xte{t} = X{t}(te_idx, :);
        cv_Yte{t} = Y{t}(te_idx, :);
    end
    
    perf_vec = zeros(length(lambda_range), 1);
    for lambda_idx = 1:length(lambda_range)
        curr_rmse = method(cv_Xtr, cv_Ytr, cv_Xte, cv_Yte, lambda_range(lambda_idx), method_opts);
        perf_vec(lambda_idx) = perf_vec(lambda_idx) + curr_rmse(end);
    end
end
perf_vec = perf_vec ./ cv_fold;
fprintf(']\n')
    
[best_rmse, index] = min(perf_vec);
best_lambda = lambda_range(index);

end