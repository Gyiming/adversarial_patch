import numpy as np
import math
import os
import sys
import cv2
import pdb
import argparse

def avg(a,i,j):
	depth = np.zeros((3))
	num = 0
	for k in range(i-10,i+10):
		for t in range(j-10,j+10):
			#check boundary
			if (i < 0):
				i = 0
			if (j < 0):
				j = 0
			if (i > 479):
				i = 479
			if (j > 639):
				j = 639
			num += 1
			depth += a[k,t]			
			#pdb.set_trace()
	#search row to find cloest depth
	'''
	r = i
	if depth[0] == 0:
		num = 1
		while(r < 479):
			if a[r,j][0] == 0:
				continue
			else:
				depth = a[r,j]
				break
	'''

	
	depth = depth/num
	depth[0] = math.ceil(depth[0])
	depth[1] = math.ceil(depth[1])
	depth[2] = math.ceil(depth[2])


	return depth

def main():
	parser = argparse.ArgumentParser(description='fill depth of the image.')
	parser.add_argument('--img',
                    help='image number to fill depth.',
                    default=740, type=int)
	args = parser.parse_args()
	img = args.img

	prev_path = './bedroom/depth_ori/' + '%06d.png'

	a = cv2.imread(prev_path % img)

	sum_0 = 0
	pdb.set_trace()
	#average depth to the non-depth pixel
	for i in range(50,430):
		for j in range(50,590):
			if a[i,j][0] == 0:
				a[i,j] = avg(a,i,j)
				#pdb.set_trace()
	#check the number of 0 depth point
	for i in range(50,430):
		for j in range(50,590):
			if a[i,j][0] == 0:
				sum_0 += 1
	print(sum_0)
	#pdb.set_trace()
	'''
	#keep finding 0 posotion until all depth has a value
	while (sum_0 != 0):
		sum_0 = 0
		for i in range(100,300):
			for j in range(100,300):
				if a[i,j][0] == 0:
					a[i,j] = avg(a,i,j)	
		#check the number of 0 depth point
		for i in range(100,300):
			for j in range(100,300):
				if a[i,j][0] == 0:
					sum_0 += 1		
		print(sum_0)	

	pdb.set_trace()
	'''
	new_path = './bedroom/depth/000' + '%06d.png'
	cv2.imwrite(new_path % img,a)

if __name__ == '__main__':
	main()