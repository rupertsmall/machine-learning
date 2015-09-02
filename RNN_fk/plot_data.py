# plot some of the fk data with predictions
# rupert small. august 2015
#

from numpy import *
from PIL import Image, ImageDraw
from neural_forward import *

MTA = genfromtxt('MT_zoneA_backup.csv', delimiter=',')
MTB = genfromtxt('MT_zoneB_backup.csv', delimiter=',')
data = genfromtxt('mini_mini_training.csv', delimiter=',')
data = data[0:10,30:]/255.
dims = shape(data)
xi1 = [96*96, 30]
xi2 = [30, 30]

counter=0
for eachline in range(0,dims[0]):
        x = data[eachline,:]
        A = neural_forward(xi1, x, MTA)[-30:]
        B = neural_forward(xi2, A, MTB)[-30:]*96.
        im = x.reshape((96,96))*255.0
        im = Image.fromarray(im).convert('RGB')
        nums = [2*j for j in range(0,15)]
        #draw = ImageDraw.Draw(im)

        #for i in nums:
        #       xcoord = B[i]
        #       ycoord = B[i+1]
        #       print str(xcoord)+', '+str(ycoord)
        #       draw.ellipse((xcoord-1,ycoord-1,xcoord+1,ycoord+1), fill='red', outline='red')

        imstring = 'fk_'+str(eachline)+'.png'
        im.save(imstring)
