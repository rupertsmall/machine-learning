# learn parameters for restricted boltzman machine
# rupert small - august 2015
#

from numpy import *

# import data
data = genfromtxt('test2.csv',delimiter=',')
digits = data[:,0]
dim = shape(data[:,1:])
X = c_[zeros([dim[0], 10]), ones(dim)*(data[:,1:] > 0)]
for i in range(0,len(digits)): X[i,digits[i]] = 1
pixels = dim[1]

#initialise variables
hidden = 625 # 25**2			# make it square so can plot filters in a square
visible = pixels + 10			# visible nodes are input values + ten for integers 0, .., 9
W = random.normal(0,1,100*visible*hidden).reshape((10*visible, 10*hidden))	# coefficients for bonds between hidden and visible nodes
a = random.normal(0,1, visible*10).reshape((visible, 10))			# coefficients for visible nodes
b = random.normal(0,1, hidden*10).reshape((hidden, 10))				# coefficients for hidden nodes
alpha = .02 		# learning rate
time = 0.0
epochs = 5
N = len(digits)

print
print sum(W)
print sum(a)
print sum(b)
print

def prob_v_with_h(h, W, a):
	la = len(a)
	probs = [1/(1. + exp(a[i] + dot(W[i,:],h))) for i in range(0, la)]
	n = [random.binomial(1, probs[i]) for i in range(0,la)]
	return array(n)[:, newaxis].T

def prob_h_with_v(v, W, b):
	lb = len(b)
	probs = [1/(1. + exp(b[i] + dot(v,W[:,i]))) for i in range(0, lb)]
	n = [random.binomial(1, probs[i]) for i in range(0,lb)]
	return array(n)[:, newaxis]

while time <= N*epochs:

	row = random.randint(0, dim[0])
	digit = digits[row]
	index = [digit*visible, digit*hidden]
	local_W = W[index[0]:index[0]+visible, index[1]: index[1]+hidden]
	local_a = a[:, digit]
	local_b = b[:, digit]
	v = X[row,:][:, newaxis].T
	h = prob_h_with_v(v, local_W, local_b)
	positive = dot(v.T,h.T)
	v2 = prob_v_with_h(h, local_W, local_a)
	h2 = prob_h_with_v(v2, local_W, local_b)
	negative = dot(v2.T,h2.T)
	local_W -= alpha*(positive - negative)
	W[index[0]:index[0]+visible, index[1]: index[1]+hidden] = local_W
	a[:, digit] -= alpha*(v-v2)[0,:]
	b[:, digit] -= alpha*(h-h2)[:,0] 
	
	time += 1
	if time % 100 == 0:
		print 'local_W sum: '+ str(int(sum(local_W))) + '\tsum(a): '+ str(int(sum(a))) + '\tsum b: '+str(int(sum(b)))+'\ttime: '+ format(time*100/(N*epochs), '.2f') + '\tpos - neg sum: '+ str(sum(positive - negative))
	if time % 1000 == 0:
		savetxt('boltzman_W.csv', W, delimiter=',')
		savetxt('boltzman_a.csv', a, delimiter=',')
		savetxt('boltzman_b.csv', b, delimiter=',')


savetxt('boltzman_W.csv', W, delimiter=',')
savetxt('boltzman_a.csv', a, delimiter=',')
savetxt('boltzman_b.csv', b, delimiter=',')
