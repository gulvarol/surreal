# Example usage:
# python smpl_relations.py \
# --fileinfo /home/gvarol/datasets/SURREAL/data/cmu/train/run0/01_01/01_01_c0001_info.mat \
# --t_beg 0 --t_end 100

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import sys
import transforms3d

SMPL_PATH = os.getenv('SMPL_PATH', '../../')
sys.path.append(SMPL_PATH)
from smpl_webuser.serialization import load_model


def joint_names():
    return ['hips',
            'leftUpLeg',
            'rightUpLeg',
            'spine',
            'leftLeg',
            'rightLeg',
            'spine1',
            'leftFoot',
            'rightFoot',
            'spine2',
            'leftToeBase',
            'rightToeBase',
            'neck',
            'leftShoulder',
            'rightShoulder',
            'head',
            'leftArm',
            'rightArm',
            'leftForeArm',
            'rightForeArm',
            'leftHand',
            'rightHand',
            'leftHandIndex1',
            'rightHandIndex1']


def draw_joints2D(joints2D, ax=None, kintree_table=None, with_text=True, color='g'):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints2D[j1, 0], joints2D[j2, 0]],
                [joints2D[j1, 1], joints2D[j2, 1]],
                color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_text:
            ax.text(joints2D[j2, 0],
                    joints2D[j2, 1],
                    s=joint_names()[j2],
                    color=color,
                    fontsize=8)


def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec


# Returns intrinsic camera matrix
# Parameters are hard-coded since all SURREAL images use the same.
def get_intrinsic():
    # These are set in Blender (datageneration/main_part1.py)
    res_x_px = 320  # *scn.render.resolution_x
    res_y_px = 240  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
    return K


# Returns extrinsic camera matrix
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv


def project_vertices(points, intrinsic, extrinsic):
    homo_coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose()
    proj_coords = np.dot(intrinsic, np.dot(extrinsic, homo_coords))
    proj_coords = proj_coords / proj_coords[2]
    proj_coords = proj_coords[:2].transpose()
    return proj_coords


def get_frame(filevideo, t=0):
    cap = cv2.VideoCapture(filevideo)
    cap.set(propId=1, value=t)
    ret, frame = cap.read()
    frame = frame[:, :, [2, 1, 0]]
    return frame


def main():
    parser = argparse.ArgumentParser(description='Demo to read SMPL vertices of the SURREAL dataset.')
    parser.add_argument('--fileinfo', type=str,
                        help='Path to the *_info.mat file')
    parser.add_argument('--t_beg', type=int, default=0,
                        help='Frame number (default 0)')
    parser.add_argument('--t_end', type=int, default=1,
                        help='Frame number (default 1)')
    args = parser.parse_args()
    print('fileinfo: {}'.format(args.fileinfo))
    print('t_beg: {}'.format(args.t_beg))
    print('t_end: {}'.format(args.t_end))

    info = sio.loadmat(args.fileinfo)

    # <========= LOAD SMPL MODEL BASED ON GENDER
    if info['gender'][0] == 0:  # f
        m = load_model(os.path.join(SMPL_PATH, 'models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'))
    elif info['gender'][0] == 1:  # m
        m = load_model(os.path.join(SMPL_PATH, 'models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'))
    # =========>

    root_pos = m.J_transformed.r[0]

    zrot = info['zrot']
    zrot = zrot[0][0]  # body rotation in euler angles
    RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                       (math.sin(zrot), math.cos(zrot), 0),
                       (0, 0, 1)))
    intrinsic = get_intrinsic()
    extrinsic, R, T = get_extrinsic(info['camLoc'])

    plt.figure(figsize=(18, 10))
    for t in range(args.t_beg, args.t_end):

        joints2D = info['joints2D'][:, :, t].T
        joints3D = info['joints3D'][:, :, t].T
        pose = info['pose'][:, t]
        pose[0:3] = rotateBody(RzBody, pose[0:3])
        # Set model shape
        m.betas[:] = info['shape'][:, 0]
        # Set model pose
        m.pose[:] = pose
        # Set model translation
        m.trans[:] = joints3D[0] - root_pos

        smpl_vertices = m.r
        smpl_joints3D = m.J_transformed.r

        # Project 3D -> 2D
        proj_smpl_vertices = project_vertices(smpl_vertices, intrinsic, extrinsic)
        proj_smpl_joints3D = project_vertices(smpl_joints3D, intrinsic, extrinsic)
        proj_joints3D = project_vertices(joints3D, intrinsic, extrinsic)

        # Read frame of the video
        rgb = get_frame(args.fileinfo[:-9] + '.mp4', t)
        plt.clf()
        # Show 2D skeletons (note that left/right are swapped)
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        # Released joints2D variable
        draw_joints2D(joints2D, ax1, m.kintree_table, color='b')
        # SMPL 3D joints projection
        draw_joints2D(proj_smpl_joints3D, ax1, m.kintree_table, color='r')
        # Released joints3D variable projection
        draw_joints2D(proj_joints3D, ax1, m.kintree_table, color='g')
        # Show vertices projection
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.scatter(proj_smpl_vertices[:, 0], proj_smpl_vertices[:, 1], 1)
        plt.pause(1)
    plt.show()


if __name__ == '__main__':
    main()
