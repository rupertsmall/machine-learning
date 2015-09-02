# run trained network on test data
# rupert small, august 2015
#

from numpy import *
from neural_forward import *

data = genfromtxt('test.csv', delimiter =',')[:,1:]/255.
dims = shape(data)
MTZ1 = genfromtxt('MT_zoneA.csv', delimiter=',')
MTZ2 = genfromtxt('MT_zoneB.csv', delimiter=',')
lookup_table = open('IdLookupTable.csv', 'rb')
xi1 = [96*96, 30]
xi2 = [30, 30]
headings = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']

print 'RowId,Location'
for i in range(0,dims[0]):
        x = data[i,:]
        z1out = neural_forward(xi1, x, MTZ1)
        z2in = z1out[-30:]
        results = neural_forward(xi2, z2in, MTZ2)[-30:]
        lookup_iterator = open('IdLookupTable.csv', 'rb')
        for eachline in lookup_iterator:
                values = eachline.split(',')
                row_id = int(values[0])
                image_id = int(values[1])
                feature_name = values[2].rstrip('\r\n')

                if image_id == i+1:
                        get_value = headings.index(feature_name)
                        print str(row_id) + ',' + str(96*results[get_value])
