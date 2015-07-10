% back prop algorithm
 
function [DELTA] = back_propagation(y, A, MEGA_THETA, xi)
 
L = length(xi);
a = A(end-xi(end)+1: end);           % output layer from forward prop
delta = a - y;                       % output layer error
[Dm,Dn] = size(MEGA_THETA);          % grab dimensions
DELTA = zeros(Dm, Dn);               % matrix of dJ/dtheta
rowsA = sum(xi)+L-1;
 
% indices for matrices
end_index = rowsA - xi(end);
start_index = end_index - xi(end-1);
D_row_start = Dm - xi(end) + 1; 
D_row_end = Dm;
D_col_start = Dn - xi(end-1);
D_col_end = Dn;
 
% do first step outside for loop (doesn't involve MEGA_THETA)
a_prev = a;                         % save for later
a = A(start_index:end_index);       % next a (left) into the layers
DELTA(D_row_start:D_row_end, D_col_start:D_col_end) = kron(delta, a');
 
for i = L-1:-1:2
    
    % calculate next delta
    g = a_prev.*(1-a_prev);
    local_theta = MEGA_THETA(D_row_start:D_row_end, D_col_start:D_col_end);
    delta = local_theta'*(g.*delta);
    delta = delta(2:end);
    
    % update DELTA indices for next loop
    D_row_end = D_row_start - 1;
    D_row_start = D_row_end - xi(i) + 1;
    D_col_end = D_col_start - 1;
    D_col_start = D_col_end - xi(i-1);
 
    % update indices to select from A
    end_index = start_index - 1;
    start_index = end_index - xi(i-1);
    
    % calculate next layer of DELTA
    a_prev = a(2:end);     % save for next loop
    a = A(start_index:end_index);
    
    DELTA(D_row_start:D_row_end, D_col_start:D_col_end) = kron(delta, a');
    
end
 
end
