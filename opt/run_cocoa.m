function [rmse, primal_objs, dual_objs, max_its] = run_cocoa(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
% CoCoA Method - Same Theta
% Inputs
%   Xtrain: input training data
%   Ytrain: output training data
%   Xtest: input test data
%   Ytest: output test data
%   lambda: regularization parameter
%   opts: optional arguments
% Output
%   Average RMSE across tasks, primal and dual objectives

%% intialize variables
fprintf('Running CoCoA\n');
m = length(Xtrain); % # of tasks
d = size(Xtrain{1}, 2); % # of features
W = zeros(d, m); alpha = cell(m,0);
Sigma = eye(m) * (1/m);
Omega = inv(Sigma);
totaln = 0; n = zeros(m, 1);
for t = 1:m
    n(t) = length(Ytrain{t});
    totaln = totaln + n(t);
    alpha{t} = zeros(n(t), 1);
end

%% intialize counters
rho = 1.0;
rmse = zeros(opts.cocoa_inner_iters, 1);
dual_objs = zeros(opts.cocoa_inner_iters, 1); 
primal_objs = zeros(opts.cocoa_inner_iters, 1);
max_its = zeros(opts.cocoa_inner_iters, 1);
max_cocoa_iters = 1000;

for h = 1:opts.cocoa_outer_iters
    % update W
    for hh = 1:opts.cocoa_inner_iters
        rng(hh * 1000)
        
        % compute RMSE
        curr_rmse = compute_rmse(Xtest, Ytest, W, opts);
        rmse(hh) = curr_rmse;
        primal_objs(hh) = compute_primal(Xtrain, Ytrain, W, Omega, lambda);
        dual_objs(hh) = compute_dual(alpha, Ytrain, W, Omega, lambda);

        % loop over tasks (in parallel)
        deltaW = zeros(d, m);
        deltaB = zeros(d, m);
        curr_iters = zeros(m, 1);
        for t = 1:m
            % run SDCA locally
            tperm = randperm(n(t));
            alpha_t = alpha{t};
            curr_sig = Sigma(t,t);
            curr_theta = 1000;
            i = 1;
            while (curr_theta > opts.theta && i < max_cocoa_iters)
                % select random coordinate
                idx = tperm(mod(i, n(t)) + 1);
                alpha_old = alpha_t(idx);
                curr_y = Ytrain{t}(idx);
                curr_x = Xtrain{t}(idx, :);
                
                % compute update
                update = (curr_y * curr_x * (W(:,t) + rho * deltaW(:, t)));
                grad = lambda * n(t) * (1.0 - update) / (curr_sig * rho * (curr_x * curr_x')) + (alpha_old * curr_y);
                alpha_t(idx) = curr_y * max(0.0, min(1.0, grad));
                currdW = Sigma(t, t) * (alpha_t(idx) - alpha_old) * curr_x' / (lambda * n(t));
                deltaW(:, t) = deltaW(:, t) + currdW;
                deltaB(:, t) = deltaB(:, t) + (alpha_t(idx) - alpha_old) * curr_x' / n(t);
                alpha{t} = alpha_t;
            
                if(mod(i,5) == 1 && max(curr_iters) < max_cocoa_iters)
                    curr_theta = compute_local_gap(Xtrain{t}, alpha{t}, ... 
                    Ytrain{t}, W(:, t) + deltaW(:, t), Omega(t,t), ... 
                    lambda, rho, totaln, Sigma(t,t));
                end
                i = i+1;
            end
            curr_iters(t) = i;
        end
        
        % combine updates globally
        for t = 1:m
            for tt = 1:m
                W(:, t) = W(:, t) + deltaB(:, tt) * Sigma(t, tt) * (1.0 / lambda);
            end
        end
        max_its(hh) = max(curr_iters);
        
    end
    
    %% make sure eigenvalues are positive
    A = W'*W;
    if(any(eig(A) < 0))
        [V,Dmat] = eig(A);
        dm= diag(Dmat);
        dm(dm <= 1e-7) = 1e-7;
        D_c = diag(dm);
        A = V*D_c*V';
    end
    
    %% update Omega, Sigma
    sqm = sqrtm(A);
    Sigma = sqm / trace(sqm);
    Omega = inv(Sigma);
    rho = max(sum(abs(Sigma),2)./ diag(Sigma));

end

end