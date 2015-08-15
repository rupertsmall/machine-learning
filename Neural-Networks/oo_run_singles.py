#
# optimise a neural network for image recognition
# rupert small, august 2015
#

from numpy import genfromtxt
from oo_dr_singles import *
import threading

# initiate data
data = genfromtxt('train2.csv', delimiter=',')
num_cpus = 30		# multi-threading
y_vals = data[:,0]	# outputs
x_vals = data[:,1:].T	# inputs
alpha = 100*num_cpus**2
beta = .002
base_in = 783
input_layer = 784**2
mangle_upper = 35
xi = array([input_layer,2])
threads = []

# initiate classes
zero = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 0)
one = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 1)
two  = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 2)
three = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 3)
four = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 4)
five = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 5)
six  = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 6)
seven = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 7)
eight = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 8)
nine = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 9)

# run
thr = threading.Thread(target=zero.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=one.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=two.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=three.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=four.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=five.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=six.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=seven.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=eight.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)
thr = threading.Thread(target=nine.optimise_network, args=(xi, x_vals, y_vals))
threads.append(thr)

for i in range(0,len(threads)):
	threads[i].start()

