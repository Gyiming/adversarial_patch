import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
import numpy as np
import math
mpl.rcParams['legend.numpoints'] = 1
plt.rcParams["font.family"] = "Arial"
import os
import sys
import cv2
import datetime
import time
import threading

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    #points = points[:, :3]  # exclude luminance
    return points


def output2ply(points, outName):

    outFile = open(outName, 'w')
    outFile.write("ply\n")
    outFile.write("format ascii 1.0\n")
    outFile.write("element vertex %d\n" % (len(points)))
    outFile.write("property float x\n")
    outFile.write("property float y\n")
    outFile.write("property float z\n")
    outFile.write("end_header\n")
    for i in range(len(points)):
        outFile.write("%f %f %f\n" % (points[i,0], points[i,1], points[i,2]))
    outFile.close()

def load_calib(calib_dir, velo_calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    word_dict = {}
    for line in lines:
        comps = line.split()
        word_dict[comps[0]] = comps[1:]

    lines = open(velo_calib_dir).readlines()
    R_line = lines[1].split()[1:]
    T_line = lines[2].split()[1:]
    velo_line = []
    for i in range(3):
        velo_line += R_line[i*3:3+i*3]
        velo_line.append(T_line[i])

    P = np.array(word_dict["P_rect_02:"]).reshape(3,4)
    #
    Tr_velo_to_cam = np.array(velo_line).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(word_dict["R_rect_02:"][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices ,:]
    pts3d[:,3] = 1
    return pts3d.transpose(), indices

def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    # print(pts3d_cam.shape)
    if (pts3d_cam[2]<0): return None
    pts2d_cam = Prect.dot(pts3d_cam)
    return pts2d_cam/pts2d_cam[2]

def update_moved_img(moved_img, org_img, depth, moves, distance, prev_c, prev_r, curr_c ,curr_r):

	path_size = 1
	rows, cols = org_img.shape[:2]
	for i in range(-path_size, path_size+1):
		for j in range(-path_size, path_size+1):
			r1 = prev_r + i 
			c1 = prev_c + j
			r2 = curr_r + i 
			c2 = curr_c + j

			if (c1 < cols and r1 < rows and r1 > 0 and c1 > 0 
				and c2 < cols and r2 < rows and r2 > 0 and c2 > 0):

				if depth[r2, c2] > distance or depth[r2, c2] == 0:
					depth[r2, c2] = distance
					moved_img[r2, c2]= org_img[r1, c1]
					moves[r2, c2, 0] = curr_r - prev_r
					moves[r2, c2, 1] = curr_c - prev_c

	return moved_img, depth, moves

def find_nearest_move(moved_img, org_img, depth, moves, c, r):
	rows, cols = org_img.shape[:2]

	offsets = []
	for i in range(1, cols):
		
		if ((c+i) < cols and r < rows and r > 0 and c > 0 and depth[r, c+i] > 0):
			offsets.append(moves[r, c+i])

		if (c < cols and r < rows and r > 0 and (c-i) > 0 and depth[r, c-i] > 0):
			offsets.append(moves[r, c-i])

		if (c < cols and (r+i) < rows and r > 0 and c > 0 and depth[r+i, c] > 0):
			offsets.append(moves[r+i, c])

		if (c < cols and r < rows and (r-i) > 0 and c > 0 and depth[r-i, c] > 0):
			offsets.append(moves[r-i, c])

		if len(offsets) > 8:
			offsets = np.array(offsets)
			mean_r = np.mean([offset[0] for offset in offsets])
			mean_c = np.mean([offset[1] for offset in offsets])
			old_r = r - int(mean_r)
			old_r = max(0, min((rows - 1), old_r))
			old_c = c - int(mean_c)
			old_c = max(0, min((cols - 1), old_c))
			moved_img[r, c] = org_img[old_r, old_c].copy()
			moves[r, c, 0] = mean_r
			moves[r, c, 1] = mean_c
			# depth[r, c] = 1
			return moved_img, moves, depth

	return moved_img, moves, depth



def align_img_and_pc(img_dir, pc_dir, calib_dir, velo_calib_dir, vehicle_move, rot_matx):

	org_img = cv2.imread(img_dir)
	org_img_copy = org_img.copy()
	pts = load_velodyne_points(pc_dir)
	P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir, velo_calib_dir)

	pts3d, indices = prepare_velo_points(pts)
	pts3d_ori = pts3d.copy()


	rows, cols = org_img.shape[:2]

	depth = np.zeros_like(org_img[:,:,0]).astype(int)
	moves = np.zeros_like(org_img[:,:,0:2]).astype(int)
	moved_img = np.zeros_like(org_img)

	for idx in range(len(pts3d[1])):
		pt = pts3d[:,idx]
		# find normal
		pts2d_normed = project_velo_points_in_img(pt, Tr_velo_to_cam, R_cam_to_rect, P)

		if pts2d_normed is None:
			continue
		prev_c = int(np.round(pts2d_normed[0]))
		prev_r = int(np.round(pts2d_normed[1]))


		# update value
		pt[0] -= vehicle_move
		pt = rot_matx.dot(pt.transpose())
		pt = pt.transpose()

		pts2d_normed = project_velo_points_in_img(pt, Tr_velo_to_cam, R_cam_to_rect, P)

		if pts2d_normed is None:
			continue

		curr_c = int(np.round(pts2d_normed[0]))
		curr_r = int(np.round(pts2d_normed[1]))

		if (prev_c < cols and prev_r < rows and prev_r > 0 and prev_c > 0
			and curr_c < cols and curr_r < rows and curr_r > 0 and curr_c > 0):

			distance = math.sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2])

			moved_img, depth, moves = update_moved_img(moved_img, org_img, depth, moves, distance, prev_c, prev_r, curr_c ,curr_r)

			# moved_img[curr_r:curr_r+8, curr_c:curr_c+8] = org_img[prev_r:prev_r+8, prev_c:prev_c+8]

	'''
	for r in range(rows-1, -1, -1):
		for c in range(cols):
			if depth[r, c] == 0:
				moved_img, moves, depth = find_nearest_move(moved_img, org_img_copy, depth, moves, c, r)
	'''

	return moved_img, moves
	

def load_speed_logs(oxt_dir, n_frames):
	vehicle_speed = []
	vehicle_rads = []

	for i in range(n_frames):
		oxt_path = oxt_dir + "%010d.txt" % i
		with open(oxt_path, "r") as f:
			contents = f.read()
			contents = contents.split(" ")
			vehicle_speed.append(float(contents[8]))
			vehicle_rads.append(float(contents[19]))

	return vehicle_speed, vehicle_rads

def load_timestamp(timestamp_path, n_frames):
	timestamps = []
	with open(timestamp_path) as f:
		for line in f.readlines():
			timestamps.append(datetime.datetime.strptime(line.rstrip()[:-3], '%Y-%m-%d %H:%M:%S.%f'))

	return timestamps

dir_name = "dataset/2011_09_30/2011_09_30_drive_0018_sync/"
calib_dir = "dataset/2011_09_30/calib_cam_to_cam.txt"
velo_calib_dir  = "dataset/2011_09_30/calib_velo_to_cam.txt"

timestamp_file = "oxts/timestamps.txt"
oxt_dir = "oxts/data/"
pc_dir = "velodyne_points/data/"
img_dir = "image_02/data/"

n_frames = 2762

vehicle_speed, vehicle_rads = load_speed_logs(dir_name+oxt_dir, n_frames)

timestamps = load_timestamp(dir_name+timestamp_file, n_frames)

def plotPerFrameSpeed(vehicle_speed):

    x_axis = range(len(vehicle_speed))

    plt.rc('font', size=18)
    ax = plt.figure(figsize=(12, 4)).add_subplot(111)
    plt.setp(ax.spines.values(), linewidth=1.5)

    ax.grid(color='grey', which='major', axis='y', linestyle='--')
    ax.set_ylabel('Speed (m/s)', fontsize=20)
    ax.set_xlabel('Frame No.', fontsize=20)


    p1 = ax.plot(x_axis, vehicle_speed, linestyle='-', linewidth=2)

    plt.savefig("vehicle_speed.pdf");

def plotPerFrameRad(vehicle_rads):

    x_axis = range(len(vehicle_rads))

    plt.rc('font', size=18)
    ax = plt.figure(figsize=(12, 4)).add_subplot(111)
    plt.setp(ax.spines.values(), linewidth=1.5)

    ax.grid(color='grey', which='major', axis='y', linestyle='--')
    ax.set_ylabel('Angular Speed(Rad/s)', fontsize=20)
    ax.set_xlabel('Frame No.', fontsize=20)


    p1 = ax.plot(x_axis, vehicle_rads, linestyle='-', linewidth=2)

    plt.savefig("vehicle_rads.pdf");

def generate_img(start, end):
	for i in range(start, end):
		pc_path = dir_name + pc_dir + '%010d.bin' % (i)
		img_path = dir_name + img_dir + '%010d.png' % (i)
		pc = load_velodyne_points(pc_path)

		# vehicle_move = vehicle_speed[i]*((timestamps[i+1] - timestamps[i]).total_seconds())
		vehicle_move = 0.5*(vehicle_speed[i] + vehicle_speed[i+1])*((timestamps[i+1] - timestamps[i]).total_seconds())
		vehicle_rad  = 0.5*(vehicle_rads[i] + vehicle_rads[i+1])*((timestamps[i+1] - timestamps[i]).total_seconds())

		rot_matx = np.array([[math.cos(vehicle_rad), math.sin(vehicle_rad), 0, 0],
					  		 [-math.sin(vehicle_rad), math.cos(vehicle_rad), 0, 0],
					  		 [0, 0, 1, 0], [0, 0, 0, 1]])

		start_time = time.time()

		moved_img, moves = align_img_and_pc(img_path, pc_path, calib_dir, velo_calib_dir, vehicle_move, rot_matx)

		img_path = dir_name + "moved_3_with_filling/" + '%010d.png' % (i+1)
		cv2.imwrite(img_path, moved_img)
		move_path = dir_name + "moved_3_with_filling/" + '%010d.npy' % (i+1)
		np.save(move_path, moves)

		print("time %.2f" % (time.time() - start_time) , "img_path", img_path)

		# output2ply(pc, "%010d.ply" % i)

# def test():
# 	num = 120
# 	for i in range(num, num+3):
# 		pc_path = dir_name + pc_dir + '%010d.bin' % (i)
# 		img_path = dir_name + img_dir + '%010d.png' % (i)
# 		pc = load_velodyne_points(pc_path)

# 		for j in range(num+3-i):
# 			vehicle_move = 0.5*(vehicle_speed[j] + vehicle_speed[j+1])*((timestamps[j+1] - timestamps[j]).total_seconds())

# 			pc[:, 0] -= vehicle_move

# 			vehicle_rad = 0.5*(vehicle_rads[i] + vehicle_rads[i+1])*((timestamps[i+1] - timestamps[i]).total_seconds())
# 			print(vehicle_rads[i])
			

# 			rot_matx = np.array([[math.cos(vehicle_rad), math.sin(vehicle_rad), 0, 0],
# 					  [-math.sin(vehicle_rad), math.cos(vehicle_rad), 0, 0],
# 					  [0, 0, 1, 0], [0, 0, 0, 1]])

# 			print(pc.transpose().shape)
# 			pc = rot_matx.dot(pc.transpose())
# 			pc = pc.transpose()

# 		start_time = time.time()

# 		# moved_img = align_img_and_pc(img_path, pc_path, calib_dir, velo_calib_dir, vehicle_move)

# 		img_path = dir_name + "moved/" + '%010d.png' % (i+1)

# 		# cv2.imwrite(img_path, moved_img)
# 		print("time %.2f" % (time.time() - start_time) , "img_path", img_path)

# 		output2ply(pc, "%010d.ply" % i)

# test()
generate_img(0, 2762)

# for i in range(n_thread):
	
# 	th = threading.Thread(target=generate_img, args=(i, n_thread,))
# 	th.start()
	




