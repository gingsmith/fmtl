%% driver for prediction error of global, local, and MTL models

%% load dataset
datarepo = 'data/'; % location of data folder
dataset = 'small'; % toy dataset
load([datarepo dataset]); % load data

%% set parameters
addpath('opt/'); addpath('util/'); % add functions
ntrials = 1; % number of trials to run
training_percent = 0.75; % percentage of data for training
opts.obj='C'; % classification
opts.avg = 1; % compute average error across tasks

%% set hyperparameter search space
lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10]; % regularizer

%% initialize
err_constant = zeros(ntrials, 1);
err_local = zeros(ntrials, 1);
err_global = zeros(ntrials, 1);
err_mtl = zeros(ntrials, 1);

%% compare global, local, and MTL over iters trials
for trial = 1:ntrials
    %% partition the data randomly
    rng(trial); 
    [Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, training_percent);

    %% constant baseline
    opts.type = 'constant';
    rmse_constant = baselines(Xtrain, Ytrain, Xtest, Ytest, 0, opts);
    err_constant(trial) = rmse_constant;

    %% global model
    opts.type = 'global';
    opts.max_sdca_iters = 500;
    opts.tol = 1e-5;
    global_lambda = cross_val_1(Xtrain, Ytrain, 'baselines', opts, lambda_range, 5); % determine via 5-fold cross val
    err_global(trial) = baselines(Xtrain, Ytrain, Xtest, Ytest, global_lambda, opts);
    
    %% local model
    opts.type = 'local';
    opts.max_sdca_iters = 500;
    opts.tol = 1e-5;
    local_lambda = cross_val_1(Xtrain, Ytrain, 'baselines', opts, lambda_range, 5); % determine via 5-fold cross val
    err_local(trial) = baselines(Xtrain, Ytrain, Xtest, Ytest, local_lambda, opts);
    
    %% MTL model (mocha)
    opts.mocha_outer_iters = 10;
    opts.mocha_inner_iters = 100;
    opts.mocha_sdca_frac = 0.5;
    opts.w_update = 0; % do a full run, not just one w-update
    opts.sys_het = 0; % not messing with systems heterogeneity
    mocha_lambda = cross_val_1(Xtrain, Ytrain, 'run_mocha', opts, lambda_range, 5); % determine via 5-fold cross val
    [rmse_mocha_reg, primal_mocha_reg, dual_mocha_reg] = run_mocha(Xtrain, Ytrain, Xtest, Ytest, mocha_lambda, opts);
    err_mtl(trial) = rmse_mocha_reg(end);
    
end

