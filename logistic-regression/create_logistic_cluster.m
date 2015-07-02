% create logistic data
 
x1 = 18*rand(5000,1) - 8;
x2 = 18*rand(5000,1) - 6;
x3 = x1.*x2;
x4 = x1.*x1;
x5 = x2.*x2;
y = x4 + x5 - 2*x1 -6*x2;
y = (y <= -1);
 
x1inside = x1(logical(y));
x2inside = x2(logical(y));
x1out = x1(logical(~y));
x2out = x2(logical(~y));
figure
plot(x1inside, x2inside,'og');
hold on
plot(x1out,x2out, '.k')
axis square
grid
