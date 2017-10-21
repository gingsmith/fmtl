function [ err ] = compute_rmse(X, Y, W, opts)
% Computes RMSE for MTL
% X: m-length cell of nxd features
% Y: m-length cell of nx1 labels
% W: dxm weight matrix
% opts:
%   opts.avg: compute avg (opts.avg = 1) or total (opts.avg = 0) rmse
%   opts.obj: 'R' for regression, 'C' for classification

%% compute predicted values
m = length(X);
Y_hat = cell(m,1);
for t=1:m
    if(opts.obj == 'R')
        Y_hat{t} = X{t} * W(:,t);
    else
        Y_hat{t} = sign(X{t} * W(:, t));
    end
end

%% compute errors
if(opts.avg)
    all_errs = zeros(m,1);
    for t=1:m
        if(opts.obj == 'R')
            all_errs(t) = sqrt(mean((Y{t} - Y_hat{t}).^2));
        else
            all_errs(t) = mean(Y{t} ~= Y_hat{t});
        end
    end 
    % compute mean error
    err = mean(all_errs);
else
    % compute total error
    Y = cell2mat(Y(:));
    Y_hat = cell2mat(Y_hat);
    if(opts.obj == 'R')
        err = sqrt(mean((Y - Y_hat).^2));
    else
        err = mean(Y ~= Y_hat);
    end
end

end

