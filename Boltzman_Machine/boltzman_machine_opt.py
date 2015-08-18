# learn paramters for restricted boltzman machine
# rupert small - august 2015
#

from numpy import *

# import data
data = genfromtxt('test2.csv',delimiter=',')
digits = data[:,0]
dim = shape(data[:,1:])
X = ones(dim)*(data[:,1:] > 0)
pixels = dim[1]

#initialise variables
hidden = 625 # 25**2			# make it square so can plot filters in a square
visible = pixels + 10			# visible nodes are input values + ten for integers 0, .., 9
DELTA_W = ones([10*visible, 10*hidden])	# coefficients for bonds between hidden and visible nodes
a = ones([visible, 10])	# coefficients for visible nodes
b = ones([hidden, 10])	# coefficients for hidden nodes
alpha = 2 		# learning rate
time = 0.0
epochs = 10
N = len(digits)

while time < N*epochs:

	row = random.randint(0, dim[0])
	digit = digits[row]
	index = [digit*visible, digit*hidden]
	local_W = DELTA_W[index[0]:index[0]+visible, index[1]: index[1]+hidden]
	v = X[row,:]
	h = prob_h_with_v(v, local_W, a, b)
	positive = kron(v, h)
	v2 = prob_v_with_h(h, local_W, a, b)
	h2 = prob_h_with_v(v2, local_W, a, b)
	negative = kron(v2,h2)
	local_W += alpha*(positive - negative)
	DELTA_W[index[0]:index[0]+visible, index[1]: index[1]+hidden] = local_W

	time += 1
	print 'time: ', (t/(N*epochs))*100
	

def prob_v_with_h(h, W, a):
	la = len(a)
	probs = [1/(1. + exp(a[i] + dot(W[i,:],h))) for i in range(0, la)]
	n = [random.binomial(1, probs[i]) for i in range(0,la)]
	return n

def prob_h_with_v(v, W, b):
	lb = len(b)
	probs = [1/(1. + exp(b[i] + dot(v,W[:,i]))) for i in range(0, len(b))]
	n = [random.binomial(1, probs[i]) for i in range(0,lb)]
	return n
