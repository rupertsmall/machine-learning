# use restricted boltzman machine filters to classify images
# rupert small - august 2015
#

from numpy import *

# import data
data = genfromtxt('train2.csv', delimiter=',')
W = genfromtxt('boltzman_W_backup.csv', delimiter=',')
a = genfromtxt('boltzman_a_backup.csv', delimiter=',')
b = genfromtxt('boltzman_b_backup.csv', delimiter=',')
dim = shape(data[:,1:])
digits = data[:,0]
X = c_[zeros([dim[0], 10]), ones(dim)*(data[:,1:] > 0)]
visible = shape(a)[0]
hidden =  shape(b)[0]
del data

def cost(v, W, h, a, b):
        v = v[:, newaxis]
        cost = -exp(-dot(a,v) - dot(b,h) - dot(dot(v.T,W),h))
        return cost

def prob_h_with_v(v, W, b):
        lb = len(b)
        probs = [1/(1. + exp(b[i] + dot(v,W[:,i]))) for i in range(0, lb)]
        n = [random.binomial(1, probs[i]) for i in range(0,lb)]
        return array(n)[:, newaxis]

success = 0
sample = 0.0
for line in range(0,dim[0]):
        v = X[line,:]
        costs = zeros(10)
        for i in range(0,10):
                local_W = W[i*visible:(i+1)*visible,i*hidden:(i+1)*hidden]
                #for j in range(0,5):   # take an average
                h = prob_h_with_v(v, local_W, b[:,i])
                costs[i] = costs[i] + cost(v, local_W, h, a[:,i], b[:,i])
                #costs = costs/5.
        predict = argmin(costs)
        if predict - digits[line] == 0: success += 1
        sample += 1
        print '\r\033[K\bsuccess: ' + str(success*100/sample),
