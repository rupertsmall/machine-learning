# initialise MEGA_THETA, the matrix determining the neural network

from numpy import *

def create_MEGA_THETA(xi):

        L = size(xi)
        sum_xi = sum(xi)
        rows = sum_xi - xi[0]
        cols = sum_xi - xi[-1] + L - 1      #  include base unit for each layer
        MEGA_THETA = random.normal(0,1,rows*cols).reshape((rows,cols))
        return MEGA_THETA
