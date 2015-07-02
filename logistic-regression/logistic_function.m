%  logistic regression
 
function [] = logistic_function(X,Y)
 
dim = size(X,2)+1;              % add ground value = 1
rowsX = size(X,1);
theta = ones(1, dim);           % model parameters
X = [ones(rowsX,1) X];
h = @(x)1/(1+exp(-theta*x'));   % logistic function
alpha = .2;                     % algorithm learning rate
error = 1;
epsilon = .001;
 
while error > epsilon
    saved_theta = theta;
    theta = theta - (alpha/dim)*((h(X) - Y')*X);
    error = sum(abs(theta-saved_theta));
end
 
end
