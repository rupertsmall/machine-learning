% find seed candidates for implementation of the K-means algorithm 
% success > 0 implies function found K distinct points with given N
 
function [Xk,Yk,success] = seed_k_means(X,Y,K,N)
 
% find candidates for mass centers
% candidate grid
xmax = max(X);
ymax = max(Y);
xmin = min(X);
ymin = min(Y);
delta_x = (xmax-xmin)/N;
delta_y = (ymax-ymin)/N;
NN = N^2;
XX = X.^2;
YY = Y.^2;
 
[x_candidates y_candidates] = meshgrid(xmin:delta_x:xmax, ymin:delta_y:ymax);
x_candidates = reshape(x_candidates, NN+2*N+1,1);
y_candidates = reshape(y_candidates, NN+2*N+1,1);
xx = x_candidates.^2;
yy = y_candidates.^2;
 
xdot = kron(X./(XX + YY), x_candidates./(xx+yy));
ydot = kron(Y./(XX + YY), y_candidates./(xx+yy));
xdist = kron(exp(X),exp(-x_candidates)).^2;
ydist = kron(exp(Y),exp(-y_candidates)).^2;
dist = ((xdist+ydist)/2 - 1).^2;
dot = xdot + ydot -1;
dot = dot.^2;
measure = dist.*dot;
measure = reshape(measure, length(x_candidates), length(X));
[closest index] = min(measure);
Xk = x_candidates(unique(index)); % the x-candidates (not necessarily length K)
Yk = y_candidates(unique(index)); % the y-candidates (not necessarily length K)
 
% success = 0 => K points found
% success > 0 => more than K points found
% success < 0 => less than K points found <=> failure
success = length(Xk) - K;
 
end
