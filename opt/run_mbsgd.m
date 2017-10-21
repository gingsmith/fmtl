function [rmse, primal_objs] = run_mbsgd(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
% Mocha Method
% Inputs
%   Xtrain: input training data
%   Ytrain: output training data
%   Xtest: input test data
%   Ytest: output test data
%   lambda: regularization parameter
%   opts: optional arguments
% Output
%   Average RMSE across tasks, primal objs

%% intialize variables
fprintf('Running Mb-SGD\n');
m = length(Xtrain); % # of tasks
d = size(Xtrain{1}, 2); % # of features
W = zeros(d, m);
Sigma = eye(m) * (1/m);
Omega = inv(Sigma);
totaln = 0; n = zeros(m, 1);
for t = 1:m
    n(t) = length(Ytrain{t});
    totaln = totaln + n(t);
end

%% intialize counters
rmse = zeros(opts.mbsgd_inner_iters, 1);
primal_objs = zeros(opts.mbsgd_inner_iters, 1);

for h = 1:opts.mbsgd_outer_iters
    %% update W
    for hh = 1:opts.mbsgd_inner_iters
        rng(hh * 1000);
        if(opts.sys_het)
            sys_iters = (opts.top - opts.bottom) .* rand(m,1) + opts.bottom;
        end
        
        % compute RMSE
        rmse(hh) = compute_rmse(Xtest, Ytest, W, opts);
        primal_objs(hh) = compute_primal(Xtrain, Ytrain, W, Omega, lambda);
        
        % loop over tasks (in parallel)
        total_loss = zeros(d, m);
        local_iters = cell(m, 0);
        for t = 1:m
            tperm = randperm(n(t));
            if(opts.sys_het)
                local_iters{t} = n(t) * sys_iters(t);
            else
                local_iters{t} = n(t) * opts.mocha_sdca_frac;
            end
            
            % compute local gradients
            for s = 1:local_iters{t}
                idx = tperm(mod(s, n(t)) + 1);
                curr_y = Ytrain{t}(idx);
                curr_x = Xtrain{t}(idx, :);
                
                % compute update
                if((curr_y * curr_x * W(:, t)) < 1.0)
                    update = curr_y .* curr_x';
                    total_loss(:, t) = total_loss(:, t) + update;
                end
            end
        end
        
        denom = sum(cell2mat(local_iters(:)));
        W = W * (eye(m) - (Omega .* (opts.mbsgd_scaling / hh)) ) + (opts.mbsgd_scaling / denom .* total_loss);
        
        % projection step [optional]
        for t=1:m
            W(:, t) = min(1.0,  1.0 / (sqrt(lambda) * norm(W(:, t), 2))) .* W(:, t);
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