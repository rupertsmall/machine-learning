%  logistic regression
 
function [theta] = logistic_function()
 
clear all
close all
 
[X Y] = create_logistic_data1(10000);   % create lots of synthetic data
 
dim = size(X,2)+1;                      % add ground value = 1
rowsX = size(X,1);
theta = ones(1, dim);                   % model parameters
X = [ones(rowsX,1) X];                  % add ground value = 1
h = @(x,y)1./(1+exp(-y*x'));            % logistic function
alpha = 1;                              % algorithm learning rate
error = 1;
epsilon = .00001;
 
while error > epsilon
    saved_theta = theta;
    theta = theta - (alpha/dim)*((h(X, theta) - Y')*X);
    error = sum(abs(theta-saved_theta));
end
 
% now normalise theta vector by the ground value coefficient
norm = theta(1);
theta = theta/norm;
% check what the model predicts now. plot should look
% identical to first one generated by create_logistic_data1()
check_logistic_output(theta,10000);
 
end
