function [ primal_obj ] = compute_primal(X, Y, W, Omega, lambda)
% Inputs
%   Xtrain: input training data (m-length cell)
%   Ytrain: output training data (m-length cell)
%   W: current models (d x m)
%   Omega: precision matrix (m x m)
%   lambda: regularization parameter
% Output
%   primal objective

% compute primal
total_loss = 0;
for t=1:length(X)
    preds = Y{t}.*(X{t}*W(:, t));
    total_loss = total_loss + mean(max(0.0, 1.0 - preds));
end
primal_obj = total_loss + lambda / 2 * trace(W * Omega * W');

end

