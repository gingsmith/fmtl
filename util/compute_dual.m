function [ dual_obj ] = compute_dual(alpha, Y, W, Omega, lambda)
% Inputs
%   alpha: dual variables (m-length cell)
%   Y: output training data (m-length cell)
%   W: current models (d x m)
%   Omega: precision matrix (m x m)
%   lambda: regularization parameter
% Output
%   primal objective

total_alpha = 0;
for tt=1:length(Y)
    total_alpha = total_alpha + mean(-1.0 .* alpha{tt} .* Y{tt});
end
dual_obj = -lambda / 2 * trace(W * Omega * W') - total_alpha;
end
