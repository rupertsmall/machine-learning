% forward propagation for neural network
 
function [output] = neural_forward(xi, input, MEGA_THETA)
 
L = length(xi);     % layers in NN (including in/out)
 
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
    % forward prop
    for i=2:L-1
        local_theta = MEGA_THETA(row_start:row_end,col_start:col_end);
        input = local_theta*input;
        row_start = row_end + 1;
        row_end = row_end + xi(i+1);
        col_start = col_end + 1;
        col_end = col_end + xi(i) + 1;
    end
    % last step due to i=2:L-1
    local_theta = MEGA_THETA(row_start:row_end,col_start:col_end);
    output = local_theta*input;
end
 
end

