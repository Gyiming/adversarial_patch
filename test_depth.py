import numpy as np
import math
import open3d as o3d
import copy

import os
import sys
import cv2
import datetime
import time
from tqdm import tqdm
import pdb
import argparse

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:,3] = 1
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

def load_transfer_info(filename):
	ransac_mats = []
	with open(filename) as f:
		lines = f.readlines()
		idx = 0
		while idx < len(lines):
			ransac_mat = []
			for i in range(1, 5):
				line = lines[idx+i]
				nums = [float(n) for n in line.rstrip().split(" ")]
				ransac_mat.append(nums)	
				
			ransac_mats.append(np.array(ransac_mat))
				
			idx += 5

	print(len(ransac_mats))

	return ransac_mats

def convert_depth_to_pcloud(depth_img, img):

	principal_point = [319.5, 239.5]
	focal_length = 525.0

	x = np.repeat([range(img.shape[1])], img.shape[0], axis=0)
	y = np.repeat([[i] for i in range(img.shape[0])], img.shape[1], axis=1)

	x_p = (x - principal_point[0])*depth_img/focal_length/1000
	y_p = (y - principal_point[1])*depth_img/focal_length/1000

	pc = []
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			pc.append([x_p[r, c], y_p[r, c], depth_img[r, c]/1000, 1])

	return np.array(pc)

def project_to_img(img, org_pc, trans_pc, scale=1):

	principal_point = [319.5, 239.5]
	focal_length = 525.0

	img_shape = img.shape
	blank_img = np.zeros((img_shape[0]//scale, img_shape[1]//scale, img_shape[2]))
	# blank_img[:,:] = [0,0,255]
	for i in range(len(org_pc)):
		org_pt = org_pc[i]
		trans_pt = trans_pc[i]

		org_x = int(org_pt[0]*focal_length/org_pt[2]+principal_point[0])
		org_y = int(org_pt[1]*focal_length/org_pt[2]+principal_point[1])

		trans_x = int(trans_pt[0]*focal_length/trans_pt[2]+principal_point[0])//scale
		trans_y = int(trans_pt[1]*focal_length/trans_pt[2]+principal_point[1])//scale
		# print(x, y)
		if (org_x >= 0 and org_y >= 0 and org_x < img.shape[1] and org_y < img.shape[0]
			and trans_x >= 0 and trans_y >= 0 and trans_x < img.shape[1]//scale and trans_y < img.shape[0]//scale):
			blank_img[trans_y, trans_x, :] = img[org_y, org_x, :]

	return blank_img


def reverse_mat(mat):
	new_mat = np.zeros_like(mat)
	for i in range(3):
		for j in range(3):
			new_mat[i, j] = mat[i, j]
			# if i == j:
			# 	new_mat[i, j] = 1
			# else:
			# 	new_mat[i, j] = 0

	new_mat[0, 3] = -mat[0, 3]
	new_mat[1, 3] = -mat[1, 3]
	new_mat[2, 3] = -mat[2, 3]
	new_mat[3, 3] = 1

	return new_mat

def read_rgbd_img(i):
	img_dir = "bedroom/image/"
	depth_dir = "bedroom/depth/"

	depth_path_regex = depth_dir + '%06d.png'
	img_path_regex = img_dir + '%06d.jpg'


	color_raw = o3d.io.read_image(img_path_regex % i)
	depth_raw = o3d.io.read_image(depth_path_regex % i)
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
	    color_raw, depth_raw)

	return rgbd_image

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(src_pcd, tgt_pcd, voxel_size):

    source_down, source_fpfh = preprocess_point_cloud(src_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(tgt_pcd, voxel_size)
    return src_pcd, tgt_pcd, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, init_trans):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def pc_registration(src_pcd, tgt_pcd):

	voxel_size = 0.05

	_, _, source_down, target_down, source_fpfh, target_fpfh = \
				prepare_dataset(src_pcd, tgt_pcd, voxel_size)

	# org_pc = np.asarray(src_pcd.points)
	# output2ply(org_pc, ply_saved_dir+'%06d.ply' % i)
	result_ransac = execute_global_registration(source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size)
	print(result_ransac)
	# draw_registration_result(source_down, target_down,
	#                          result_ransac.transformation)

	result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
	                                 voxel_size, result_ransac.transformation)

	return result_icp

def main():
	parser = argparse.ArgumentParser(description='Digital attack of adv. camera sticker.')
	parser.add_argument('--img1',
                    help='image number to fill depth.',
                    default=740, type=int)
	parser.add_argument('--img2',
                    help='image number to fill depth.',
                    default=741, type=int)
	parser.add_argument('--org',
                    help='image number to fill depth.',
                    default=True, type=bool)

	args = parser.parse_args()

	img_dir = "bedroom/image/"
	depth_dir = "bedroom/depth/"
	patch_dir = "bedroom/patch/"

	#n_frames = 20000

	depth_path_regex = depth_dir + '%06d.png'
	img_path_regex = img_dir + '%06d.jpg'
	patch_path_regex = patch_dir + '%06d.png'

	pcloud_saved_dir = 'pcloud/'

	if not os.path.exists(pcloud_saved_dir):
		os.makedirs(pcloud_saved_dir)

	ply_saved_dir = 'registered_ply/'

	if not os.path.exists(ply_saved_dir):
		os.makedirs(ply_saved_dir)

	select_img_dir = 'convert_img/'

	if not os.path.exists(select_img_dir):
		os.makedirs(select_img_dir)

	mats = load_transfer_info("pose_bedroom/bedroom.log")


	i = args.img1
	target_idx = args.img2
	org_flag = args.org

	src_image = read_rgbd_img(i)

	src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			src_image,
			o3d.camera.PinholeCameraIntrinsic(
			o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
			# Flip it, otherwise the pointcloud will be upside down
	src_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

	tgt_image = read_rgbd_img(target_idx)

	tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			tgt_image,
			o3d.camera.PinholeCameraIntrinsic(
			o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
			# Flip it, otherwise the pointcloud will be upside down
	tgt_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


	result_icp = pc_registration(src_pcd, tgt_pcd)

			# print(result_icp.transformation)

	#pdb.set_trace()

	org_pc = np.copy(np.asarray(src_pcd.points))

	src_pcd.transform(result_icp.transformation)

	trans_pc = np.asarray(src_pcd.points)

	# output2ply(np.asarray(src_pcd.points), 'src.ply')
	# output2ply(np.asarray(tgt_pcd.points), 'tgt.ply')

	#img = cv2.imread(img_path_regex % i)
	#pdb.set_trace()
	if org_flag == True:
		img = np.zeros((480, 640, 3), np.uint8)
		cv2.circle(img,(360,100),20,(0,0,255),-1)
	else:
		img = cv2.imread(patch_path_regex % i)

	#img.transform(result_icp.transformation)
	#pdb.set_trace()
	proj_img =project_to_img(img, org_pc, trans_pc)
	cv2.imwrite(patch_path_regex % target_idx, proj_img)

	img_2 = cv2.imread(img_path_regex % target_idx)
	for j in range(480):
		for k in range(640):
			if proj_img[j,k][2] != 0:
				img_2[j,k] = proj_img[j,k]
	

	print("%s/%06d_to_%06d.png" % (select_img_dir, i, target_idx))

	cv2.imwrite("%s/%06d_to_%06d.png" % (select_img_dir, i, target_idx), img_2)

if __name__ == '__main__':
	main()
