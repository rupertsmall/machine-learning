% create some synthetic logistic data

function [X Y] = create_logistic_data1(N)
 
x1 = 18*rand(N,1) - 8;
x2 = 18*rand(N,1) - 6;
x3 = x1.*x2;
x4 = x1.*x1;
x5 = x2.*x2;
 
% circle (x-1)**2 + (y-3)**2 = 9
Y = x4 + x5 - 2*x1 -6*x2;
Y = (Y <= -1);
 
x1inside = x1(logical(Y));
x2inside = x2(logical(Y));
x1out = x1(logical(~Y));
x2out = x2(logical(~Y));
 
X = [x1 x2 x3 x4 x5];
figure
plot(x1inside, x2inside,'og');
hold on
plot(x1out,x2out, '.k')
axis square
grid

end
