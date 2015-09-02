# PCA for fk data
# rupert small. september 2015
#

from numpy import *
from PIL import Image, ImageDraw

# use mini_training.csv for PCA, then mini_mini for some plots
data = genfromtxt('mini_mini_training.csv', delimiter=',')
data = data[:,30:].T
dims = shape(data)
scatter = dot(data,data.T)
take = 100

#A, V = linalg.eigh(scatter)
#sort = argsort(A)
#V = V[:,sort]
#savetxt('pca_eigenvalues.csv', V, delimiter=',')

# do after once
V = genfromtxt('pca_eigenvalues.csv', delimiter=',')
pcv = V[:,-take:]

for eachline in range(0,take):
        component = pcv[:,eachline]
        mx = max(component)
        mn = min(component)
        component = (component - mn)*255.0/mx
        im = component.reshape((96,96))
        im = Image.fromarray(im).convert('RGB')
        name = 'pca_'+str(eachline)+'.png'
        im.save(name)

for column in range(0,100):
        face = data[:,column]
        coefficients = [dot(face.T,pcv[:,i]) for i in range(0,take)]
        rep = coefficients*pcv
        estimate = sum(rep, 1)
        mx = max(estimate)
        mn = min(estimate)
        estimate = (estimate - mn)*255./mx
        im = estimate.reshape((96,96))
        im = Image.fromarray(im).convert('RGB')
        name = 'pca_face_'+str(column)+'.png'
        im.save(name)
