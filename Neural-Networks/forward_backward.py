
from numpy import *
from neural_forward import *
from back_propagation import *

def forwardBackward(xi, x, y, MT, time_queue, good_queue, DELTA_queue):
	A = neural_forward(xi, x, MT)
	check = argmax(A[-xi[-1]:])
	# send back some progress statistic
 	if y[check]-1 == 0:
		good = good_queue.get()
		good += 1
		good_queue.put(good)
		good_queue.task_done()

	time = time_queue.get()
	time += 1
	time_queue.put(time)
	time_queue.task_done()

	DELTA = DELTA_queue.get()
	DELTA = DELTA + back_propagation(y, A, MT, xi)
	DELTA_queue.put(DELTA)
	DELTA_queue.task_done()
