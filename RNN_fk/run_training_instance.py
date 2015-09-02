# run the facial keypoints classes to train a neural network
# rupert small, august 2015
# 

from numpy import *
from fk_class_definitions import *

# initialise
data = genfromtxt('training.csv',delimiter=',')
y_vals = data[:,0:30]/96.0
x_vals = data[:,30:]/255.0
inlayer = shape(x_vals)[1]
outlayer = shape(y_vals)[1]

# start everything. zone1instance will initialise and run other classes
zone1instance = fk_zone1(inlayer, outlayer)
zone1instance.optimise_network(x_vals.T, y_vals.T)
