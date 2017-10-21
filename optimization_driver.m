%% driver for statistical and systems heterogeneity experiments

%% load dataset
datarepo = 'data/'; % location of data folder
dataset = 'small'; % small test dataset
load([datarepo dataset]); % load data

%% set parameters
addpath('opt/'); addpath('util/'); % add helper functions
training_percent = 0.75; % percentage of data for training
[Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, training_percent);
opts.obj='C'; % classification
opts.avg = 1; % compute average error
opts.sys_het = 0; % run systems (1) or stats heterogeneity exps (0)
opts.top = 1.0; % highest number of rounds
opts.bottom = 0.1; % lowest number of rounds
lambda = 1e-4;

%% mocha [need to tune sdca_frac]
opts.mocha_outer_iters = 1;
opts.mocha_inner_iters = 2000;
opts.mocha_sdca_frac = 1.0;
opts.w_update = 1; % just do a single w-update
[rmse_mocha, primal_mocha, dual_mocha] = run_mocha(Xtrain, Ytrain, Xtest, Ytest, lambda, opts);

%% cocoa [need to tune sdca_frac]
opts.cocoa_outer_iters = 1;
opts.cocoa_inner_iters = 500;
opts.theta = 0.5;
[rmse_cocoa, primal_cocoa, dual_cocoa, max_its] = run_cocoa(Xtrain, Ytrain, Xtest, Ytest, lambda, opts);

%% mbsdca [need to tune sdca_frac, scaling]
opts.mbsdca_outer_iters = 1;
opts.mbsdca_inner_iters = 5000;
opts.mbsdca_sdca_frac = 0.5;
opts.mbsdca_scaling = 10;
[rmse_mbsdca, primal_mbsdca, dual_mbsdca] = run_mbsdca(Xtrain, Ytrain, Xtest, Ytest, lambda, opts);

%% mbsgd [need to tune sgd_frac, scaling]
opts.mbsgd_outer_iters = 1;
opts.mbsgd_inner_iters = 5000;
opts.mbsgd_sgd_frac = 0.5;
opts.mbsgd_scaling = 0.1;
[rmse_mbsgd, primal_mbsgd] = run_mbsgd(Xtrain, Ytrain, Xtest, Ytest, lambda, opts);

if(opts.sys_het) 
    %% plot systems heterogeneity
    % note: ensure methods have reached global optimal or enter manually
    optimal = min([primal_mocha; primal_cocoa; primal_mbsdca; primal_mbsgd]);
    
    %% calculate estimated time
    comm_cost = 100; % communication cost: Wifi=10, LTE=100, 3G=1000
    train_n = 0;
    for t=1:length(Xtrain)
         train_n = train_n + size(Xtrain{t},1);
    end

    %% calculate time based on flops and communication cost
    local_mocha_time = 8 * opts.top * train_n + comm_cost;
    local_mbsdca_time = 6 * opts.top * train_n + comm_cost;
    local_mbsgd_time = 4 * opts.top * train_n + comm_cost;
    mocha_time = 1:local_mocha_time:local_mocha_time*length(primal_mocha); 
    cocoa_time = cumsum(max_its .* (8 * length(Xtrain)) + comm_cost);
    mbsdca_time = 1:local_mbsdca_time:local_mbsdca_time*length(primal_mbsdca); 
    mbsgd_time = 1:local_mbsgd_time:local_mbsgd_time*length(primal_mbsgd); 

    %% plot results
    figure;
    step = 100;
    semilogy(mocha_time(1:step:end), primal_mocha(1:step:end) - optimal, 'LineWidth', 6)
    hold on;
    semilogy(cocoa_time(1:step:end), primal_cocoa(1:step:end) - optimal, 'LineWidth', 6)
    hold on;
    semilogy(mbsdca_time(1:step:end), primal_mbsdca(1:step:end) - optimal, 'LineWidth', 6)
    hold on;
    semilogy(mbsgd_time(1:step:end), primal_mbsgd(1:step:end) - optimal, 'LineWidth', 6)
    title([dataset ': Systems Heterogeneity'])
    xlabel('Estimated Time')
    ylabel('Primal Sub-Optimality')
    set(gca, 'fontsize', 16)
    legend({'MOCHA', 'CoCoA', 'Mb-SDCA', 'Mb-SGD'})
    axis([0 8000000 .001 100]) % set manually

else
    %% plot statistical heterogeneity
    % note: ensure methods have reached global optimal or enter manually    
    optimal = min([primal_mocha; primal_cocoa; primal_mbsdca; primal_mbsgd]);

    %% calculate time based on flops and communication cost
    train_n = 0;
    for t=1:length(Xtrain)
         train_n = train_n + size(Xtrain{t},1);
    end
    comm_cost = 100; % communication cost: Wifi=10, LTE=100, 3G=1000
    local_mocha_time = 8 * opts.mocha_sdca_frac * train_n + comm_cost;
    local_mbsdca_time = 6 * opts.mbsdca_sdca_frac * train_n + comm_cost;
    local_mbsgd_time = 4 * opts.mbsgd_sgd_frac * train_n + comm_cost;
    cocoa_time = cumsum(max_its .* (8 * length(Xtrain)) + comm_cost);
    mocha_time = 1:local_mocha_time:local_mocha_time*length(primal_mocha); 
    mbsdca_time = 1:local_mbsdca_time:local_mbsdca_time*length(primal_mbsdca);
    mbsgd_time = 1:local_mbsgd_time:local_mbsgd_time*length(primal_mbsgd);

    %% plot results
    figure;
    step = 100;
    semilogy(mocha_time(1:step:end), primal_mocha(1:step:end) - optimal, 'LineWidth', 6)
    hold on;
    semilogy(cocoa_time, primal_cocoa - optimal, 'LineWidth', 6)
    hold on;
    semilogy(mbsdca_time(1:step:end), primal_mbsdca(1:step:end) - optimal, 'LineWidth', 6)
    hold on;
    semilogy(mbsgd_time(1:step:end), primal_mbsgd(1:step:end) - optimal, 'LineWidth', 6)
    title([dataset ': Statistical Heterogeneity'])
    xlabel('Estimated Time')
    ylabel('Primal Sub-Optimality')
    set(gca, 'fontsize', 16)
    legend({'MOCHA', 'CoCoA', 'Mb-SDCA', 'Mb-SGD'})
    axis([0 7000000 .001 100]) % set manually

end
