% check logistic model output
 
 
function [] = check_logistic_output(theta,N)
 
x0 = ones(N,1);
x1 = 18*rand(N,1) - 8;
x2 = 18*rand(N,1) - 6;
x3 = x1.*x2;
x4 = x1.*x1;
x5 = x2.*x2;
 
X = [x0 x1 x2 x3 x4 x5]';
Y = theta*X;
Y = (Y <= 0);
 
x1inside = x1(logical(Y));
x2inside = x2(logical(Y));
x1out = x1(logical(~Y));
x2out = x2(logical(~Y));
 
figure
plot(x1inside, x2inside,'og');
hold on
plot(x1out,x2out, '.k')
axis square
grid
 
 
end
