import cv2
import numpy as np
import pdb

def main():
	file1 = 0
	file2 = 1
	for i in range(50):
		dir1 = 'data/000000000' + str(file1) + '.png'
		dir2 = 'data/000000000' + str(file2) + '.png'

		prev_im = cv2.imread(dir1)
		next_im = cv2.imread(dir2)

		if file1 == 0:
			tmp = prev_im
			for i in range(300,330):
				for j in range(350,380):
					tmp[i,j] = 0
		
		cv2.imwrite('a.png',tmp)
		#prvs_im = cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY)
		#next_im = cv2.cvtColor(next_im, cv2.COLOR_BGR2GRAY)
		flow = cv2.optflow.calcOpticalFlowSF(prev_im, next_im, 2, 2, 4)
		if file1 == 0:
			tmp2 = next_im
		else:
			tmp2 = cv2.imread('perturb/000000000' + str(file1) + '.png')
		for i in range(300,330):
			for j in range(350,380):
				#pdb.set_trace()
				tmp2[i+int(flow[i,j][0]),j+int(flow[i,j][1])] = 0
		file1 += 1
		file2 += 1
		adv_dir1 = 'perturb/000000000' + str(file1) + '.png'
		cv2.imwrite(adv_dir1,tmp2)
		#pdb.set_trace()
		print('flow calculated')
	
if __name__ == '__main__':
	main()