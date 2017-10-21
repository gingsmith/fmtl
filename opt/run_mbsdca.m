function [rmse, primal_objs, dual_objs] = run_mbsdca(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
% Mini-batch SDCA
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
fprintf('Running Mb-SDCA\n');
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
rmse = zeros(opts.mbsdca_inner_iters, 1);
dual_objs = zeros(opts.mbsdca_inner_iters, 1); 
primal_objs = zeros(opts.mbsdca_inner_iters, 1);

for h = 1:opts.mbsdca_outer_iters
    % update W
    for hh = 1:opts.mbsdca_inner_iters
        rng(hh * 1000);
        if(opts.sys_het)
            sys_iters = (opts.top - opts.bottom) .* rand(m,1) + opts.bottom;
        end
        
        % compute RMSE
        rmse(hh) = compute_rmse(Xtest, Ytest, W, opts);
        primal_objs(hh) = compute_primal(Xtrain, Ytrain, W, Omega, lambda);
        dual_objs(hh) = compute_dual(alpha, Ytrain, W, Omega, lambda); 
        
        % loop over tasks (in parallel)
        deltaB = zeros(d, m);
        alpha_prev = alpha;
        local_iters = cell(m, 0);
        for t = 1:m
            tperm = randperm(n(t));
            alpha_t = alpha{t};
            curr_sig = Sigma(t,t);
            if(opts.sys_het)
                local_iters{t} = n(t) * sys_iters(t);
            else
                local_iters{t} = n(t) * opts.mocha_sdca_frac;
            end
            
            % run SDCA locally
            for s=1:local_iters{t}
                % select random coordinate
                idx = tperm(mod(s, n(t)) + 1);
                alpha_old = alpha_t(idx);
                curr_y = Ytrain{t}(idx);
                curr_x = Xtrain{t}(idx, :);
                
                % compute update
                update = (curr_y * curr_x * W(:,t));
                grad = lambda * n(t) * (1.0 - update) / (curr_sig * (curr_x * curr_x')) + (alpha_old * curr_y);
                alpha_t(idx) = curr_y * max(0.0, min(1.0, grad));
                deltaB(:, t) = deltaB(:, t) + (alpha_t(idx) - alpha_old) * curr_x' / n(t);
                alpha{t} = alpha_t;
            end
        end
        
        scaling = opts.mbsdca_scaling / sum(cell2mat(local_iters(:)));
        % combine updates globally
        for t = 1:m
            alpha{t} = alpha_prev{t} + scaling * (alpha{t} - alpha_prev{t});
            for tt = 1:m
                W(:, t) = W(:, t) + scaling * deltaB(:, tt) * Sigma(t, tt) * (1.0 / lambda);
            end
        end
            
        
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

end

end