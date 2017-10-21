function [ gap ] = compute_local_gap(X, alpha, Y, W, Omega, lambda, rho, totaln, Sigma)
% Inputs
%   alpha: dual variables (m-length cell)
%   Y: output training data (m-length cell)
%   W: current models (d x m)
%   Omega: precision matrix (m x m)
%   lambda: regularization parameter
% Output
%   local primal-dual gap

n = length(Y);
dual_loss = mean(-1.0 .* alpha .* Y);
Xalph = X' * alpha;
dual_reg = (1.0 / n * (W' * Xalph)) + (rho / (2 * lambda * n * n) * Sigma * Sigma * (Xalph' * Xalph));
dual_obj = - dual_reg - dual_loss;
preds = Y.*(X*W);
primal_loss = 1 / (totaln) * sum(max(0.0, 1.0 - preds));
primal_reg = lambda / (rho * 2) * Omega * Omega * (W' * W);
primal_obj = primal_loss + primal_reg;
gap = primal_obj - dual_obj;
end