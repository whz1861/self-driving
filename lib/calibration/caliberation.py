#!/usr/bin/env python2.7
#coding=utf-8
# Baidu RPC - A framework to host and access services throughout Baidu.
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved

"""
caliberate
"""

import numpy as np
import cv2
import glob
import os
import sys
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def caliberate(input_dir, output_dir, scale=1):
    
    DEBUGMODE = 0
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(os.path.join(input_dir, '*.jpg'))
    print images
    for index, fname in enumerate(images):
        print('~~~~~~~~~~~~~~chessboard[{}]:{}~~~~~~~~~~~~~~'.format(index, fname))
        #load image
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners, fix chess corners (9, 6)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(21,21),(-1,-1),criteria)
            imgpoints.append(corners)
        
            if DEBUGMODE:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(gray, (9,6), corners,ret)
                cv2.namedWindow('img', 0)
                cv2.imshow('img',gray)
                cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    objpoints = np.asarray([objpoints], dtype='float64').reshape(-1, 1, 54, 3)
    imgpoints = np.asarray([imgpoints], dtype='float64').reshape(-1, 1, 54, 2)
    
    mtx = np.eye(3)
    dist = np.zeros(4)
    calib_flags = cv2.fisheye.CALIB_FIX_SKEW + \
                  cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                  cv2.fisheye.CALIB_CHECK_COND + \
                  cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    rvecs = np.asarray(
        [[[np.zeros(3).tolist() for i in xrange(objpoints.shape[0])]]],
        dtype='float64').reshape(-1,1,1,3)
    tvecs = np.asarray(
        [[[np.zeros(3).tolist() for i in xrange(objpoints.shape[0])]]],
        dtype='float64').reshape(-1,1,1,3)
    
    # define camera internal/external/undistort coefs
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(4)
    camera_matrix = np.eye(3)
    camera_matrix[0][0]=1000
    camera_matrix[1][1]=1000
    camera_matrix[0][2]=(gray.shape[::-1][0]-1)/2
    camera_matrix[1][2]=(gray.shape[::-1][1]-1)/2
    mtx = camera_matrix
    dist = dist_coeffs

    ret, mtx, dist, rvecs, tvecs = \
        cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], mtx, dist, rvecs, tvecs,
                              flags=calib_flags)

    print('camera_matrix = {}'.format(camera_matrix))
    print('mtx = {}'.format(mtx))
    if DEBUGMODE:
        print gray.shape[::-1]
        print('ret = {}'.format(ret))
        print('mtx = {}'.format(mtx))
        print('dist = {}'.format(dist))
        print('rvecs = {}'.format(rvecs))
        print('tvecs = {}'.format(tvecs))
        
    for index, fname in enumerate(images):
        print('~~~~~~~~~~~~~~~~~undistort[{}]:{}~~~~~~~~~~~~'.format(index, fname))
        img = cv2.imread(fname)
        #img = cv2.resize(img, (2592, 1944), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, None, None, 1.0 / scale, 1.0 / scale)
        h,  w = img.shape[:2]
        

            
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        print('newcameramtx = {}'.format(newcameramtx))
    
        nk = mtx.copy()
        nk = mtx/scale
        nk[0, 0] = mtx[0, 0] / scale
        nk[1, 1] = mtx[1, 1] / scale
        nk[2,2] = mtx[2,2]
        new_nk = nk.copy();
        new_nk[0,0] = nk[0,0]/1.5
        new_nk[1,1] = new_nk[1,1]/1.5
        print('new_nk = {}'.format(new_nk))
        undistort_mode = 0
        if undistort_mode == 0:
            mapx,mapy = cv2.fisheye.initUndistortRectifyMap(nk, dist, np.eye(3), new_nk, (w,h), 5)
            #mapx, mapy = cv2.fisheye.initUndistortRectifyMap(nk, dist, np.eye(3),
            #                                                newcameramtx, (w,h), 5)
            np.save(os.path.join(output_dir, 'mapx.npy'), mapx)
            np.save(os.path.join(output_dir, 'mapy.npy'), mapy)
            np.save(os.path.join(output_dir, 'nk.npy'), nk)
            np.save(os.path.join(output_dir, 'new_nk.npy'), new_nk)
            np.save(os.path.join(output_dir, 'dist_coefs.npy'), dist)
            cv2.imwrite(os.path.join(output_dir, 'mapx.jpg'), mapx)
            cv2.imwrite(os.path.join(output_dir, 'mapy.jpg'), mapy)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
            #dst = cv2.fisheye.undistortImage(img,mtx,dist,newcameramtx)
            #dst = cv2.resize(dst, (320, 240), interpolation=cv2.INTER_NEAREST)
            # crop the image
            x, y, w, h = roi
            #dst = dst[y:y+h, x:x+w]
        else:
            dst = cv2.undistort(img, mtx, dist, None, None)
            #dst = dst[-100:-100+h, -100:-100+w]
            # crop the image
            x, y, w, h = roi
            #dst = dst[y:y+h, x:x+w]
        
        save_name = os.path.join(output_dir, "caliberate_{}".format(os.path.basename(fname)))
        cv2.imwrite(save_name, dst)
        cv2.namedWindow('img', 0)
        cv2.namedWindow('dst', 0)
        cv2.imshow('img', img)
        cv2.imshow('dst', dst)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def undistort(input_dir, output_dir, cfg_path):
    
    if not os.path.exists(os.path.join(cfg_path, "mapx.npy")):
        print('load mapx.np error, return!!!!!')
        return
    
    mapx = np.load(os.path.join(cfg_path, "mapx.npy"))
    mapy = np.load(os.path.join(cfg_path, "mapy.npy"))
    
    
    for fname in glob.glob(os.path.join(input_dir, "*.jpg")):
        print('~~~~~~~~~~~~~~~~~~~~{}~~~~~~~~~~~~~~~~~'.format(fname))
        img = cv2.imread(fname)
        #print img.shape
        #print mapx.shape
        if img is None:
            continue
        if img.shape[:2] != mapx.shape[:2]:
            continue
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        
        save_name = os.path.join(output_dir, "caliberate_{}".format(os.path.basename(fname)))
        print save_name
        cv2.imwrite(save_name, dst)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='caliberate')
    parser.add_argument('--input_dir', dest='input_dir', help='',
                        default='./data/caliberate', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='',
                        default='./result', type=str)
    parser.add_argument('--cfg_dir', dest='cfg_dir', help='',
                        default='./result', type=str)
    parser.add_argument('--scale', dest='scale', help='',
                        default='1', type=float)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # set args
    args = parse_args()
    if args.input_dir == '':
        print(
            'Usage: python caliberate.py '
            '               --input_dir dir_of_pic'
            '               --output_dir save_to_path'
            '               --cfg_dir dir_of_cfg'
            '               --scale 4.05')
        sys.exit(1)

    caliberate(args.input_dir, args.output_dir, args.scale)
    #undistort(args.input_dir, args.output_dir, args.cfg_dir)
