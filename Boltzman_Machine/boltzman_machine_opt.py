# pseudo-code template for boltzman machine

from numpy import *

# import data
data = genfromtxt(blah)
digits = data[:,1]
X = ones(shape)

#initialise variables
hidden = 625 # 25**2			# make it square so can plot filters in a square
visible = blah + 10			# visible nodes are input values + ten for integers 0, .., 9
DELTA_W = ones([10*visible, 10*hidden])	# coefficients for bonds between hidden and visible nodes
a = ones([visible, 10])	# coefficients for visible nodes
b = ones([hidden, 10])	# coefficients for hidden nodes
alpha = 2 		# learning rate
time = 0.0
epochs = 10
N = len(digits)

while time < N*epochs:

	row = get_random_row
	digit = digits[row]
	local_W = DELTA_W[digit*visible:(digit+1)*visible, digit*hidden: (digit+1)*hidden]
	use p(h|row) to randomly select h
	positive = kron(row, h)
	use p(v|h) to randomly select a v'
	use p(h'|v') to randomly select a h'
	negative = kron(v',h')
	localW += alpha*(positive - negative)
	DELTA_W[digit*visible:(digit+1)*visible, digit*hidden: (digit+1)*hidden] = local_W

	time +=1
	print 'time: ', (t/(N*epochs))*100
	
