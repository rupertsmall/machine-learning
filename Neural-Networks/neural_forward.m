% forward propagation for neural network
 
function [output] = neural_forward(xi, input, MEGA_THETA)
 
% initialise MEGA_THETA
if nargin == 1
    MEGA_THETA = create_MEGA_THETA(xi);
    output = MEGA_THETA;    
end
 
% initialise MEGA_THETA if required, run forward prop, give result
if nargin >= 2
    if nargin == 2
        % MEGA_THETA not given, so initialise it
        MEGA_THETA = create_MEGA_THETA(xi);
    end
    % define subset of MEGA_THETA
    row_start = 1;
    row_end = xi(2);
    col_start = 1;
    col_end = xi(1) + 1;
    
    % subsets of output vectors a0, a1, a2, .. etc
    L = length(xi);     % layers in NN (including in/out)
    sumA = sum(xi) + L;
    A_start_index = 1;
    A_end_index = xi(1)+1;
    A = ones(sumA,1);
    a = input;
    A(A_start_index: A_end_index) = [1; a];
    
    % logistic function for NN nodes
    g(x) = @(x)(1./(1+exp(-x)));
 
    for i=2:L-1
        local_theta = MEGA_THETA(row_start:row_end,col_start:col_end);
        a = g(local_theta*a);
        
        A_start_index = end_index + 1;
        A_end_index = start_index + xi(i) + 1;
        row_start = row_end + 1;
        row_end = row_end + xi(i+1);
        col_start = col_end + 1;
        col_end = col_end + xi(i) + 1;
        
        A(A_start_index:A_end_index) = [1; a];
    end
    
    % last step due to i=2:L-1
    local_theta = MEGA_THETA(row_start:row_end,col_start:col_end);
    a = local_theta*a;
    % A = ; etc etc
    output = a;
end
 
end
