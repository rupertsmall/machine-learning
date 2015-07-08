% initialise MEGA_THETA, the matrix determining the neural network
 
function [MEGA_THETA] = create_MEGA_THETA(xi)
 
L = length(xi);
sum_xi = sum(xi);
rows = sum_xi - xi(1);
cols = sum_xi - xi(L) + L - 1;
% put some ones in
MEGA_THETA = eye(rows,cols);
 
end
