function [w] = simple_svm(X, y, lambda, opts)
% Simple function for solving an SVM with SDCA
% Used for local & global baselines

% Inputs
% X: input training data
% y: output training data
% lambda: regularization parameter

% Output
% w: the learned model

%% initialize
[n, d] = size(X);
w = zeros(d, 1);
alpha = zeros(n, 1);
primal_old = 0;

for iter=1:opts.max_sdca_iters
    % update coordinates cyclically
    for i=1:n
        % get current variables
        alpha_old = alpha(i);
        curr_x = X(i, :);
        curr_y = y(i);
        
        % calculate update
        grad = lambda * n * (1.0 - (curr_y * curr_x * w)) / (curr_x * curr_x') + (alpha_old * curr_y);
        
        % apply update
        alpha(i) = curr_y * max(0, min(1.0, grad));
        w = w + ((alpha(i) - alpha_old) * curr_x' * (1.0 / (lambda * n)));
    end
    
    % break if less than tol
    preds = y .* (X * w);
    primal_new = mean(max(0.0, 1.0 - preds)) + lambda * (w' * w);
    if(abs(primal_old - primal_new) < opts.tol)
        break;
    end
    primal_old = primal_new;
end

end