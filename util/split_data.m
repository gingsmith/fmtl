%% FUNCTION split_data
%   Splitting multi-task data into training / testing by percentage. 
%   
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   percent: percentage of the splitting range (0, 1)
%
%% OUTPUT
%   X_train: the split of X that has the specified percent of samples 
%   Y_train: the split of Y that has the specified percent of samples 
%   X_test: the split of X that has the remaining samples 
%   Y_test: the split of Y that has the remaining samples 
%   selIdx: the selection index of for X_train and Y_train for each task
%%

function [X_train, Y_train, X_test, Y_test, selIdx] = split_data(X, Y, percent)

if percent > 1 || percent < 0
    error('splitting percentage error')
end

task_num = length(X);

selIdx = cell(task_num, 0);
X_train = cell(task_num, 0);
Y_train = cell(task_num, 0);
X_test = cell(task_num, 0);
Y_test = cell(task_num, 0);

for t = 1:task_num
    task_sample_size = length(Y{t});
    tSelIdx = randperm(task_sample_size) < task_sample_size * percent;
    
    selIdx{t} = tSelIdx;
    
    X_train{t} = X{t}(tSelIdx,:);
    Y_train{t} = Y{t}(tSelIdx,:);
    X_test{t} = X{t}(~tSelIdx,:);
    Y_test{t} = Y{t}(~tSelIdx,:);
    
end