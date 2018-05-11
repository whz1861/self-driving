
# coding = utf-8

import os
import sys
import cv2
import numpy as np
import glob
import argparse

        
def birdEye_calib(scale=1.0, cfg_dir=None):
    """
    src_points = np.array([[1035, 1172], [1463, 1155], [1693, 1735], [860, 1762]],
                          dtype=np.float32)
    dst_points = np.array([[1035, 1172], [1465, 1172], [1465, 1860], [1035, 1860]],
                          dtype=np.float32)
    """
    src_points = np.array([[295.6, 170.2], [347.5, 170.2], [528.3, 440], [143.7, 440]],
                          dtype=np.float32)
    dst_points = np.array([[295.6, 170.2], [347.5, 170.2], [347.5, 440], [295.6, 440]],
                          dtype=np.float32)
    
    src_points = src_points * scale
    dst_points = dst_points * scale
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    np.save(os.path.join(args.cfg_dir, "birdEyeProject.npy"), M)
    np.save(os.path.join(args.cfg_dir, "birdEyeProject_inv.npy"), M_inv)
    print('M = {}'.format(M))
    print('M_inv = {}'.format(M_inv))
    
    return M, M_inv
    
def birdEye_project(input_dir, output_dir, cfg_path):
    
    M = np.load(os.path.join(cfg_path, "birdEyeProject.npy"))
    print('M = {}'.format(M))
    
    for fname in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(fname)
        
        #img = cv2.resize(img, None, None, 1./4.05, 1./4.05)
        
        if img is None:
            continue
        h, w = img.shape[:2]
        print('img.shape = {}'.format(img.shape))
        dst = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR)

        save_name = os.path.join(output_dir, os.path.basename(fname))
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
            'Usage: python birdeye.py '
            '               --input_dir dir_of_pic'
            '               --output_dir save_to_path'
            '               --cfg_dir dir_of_cfg')
        sys.exit(1)
    
    M, M_inv = birdEye_calib(1.0, args.cfg_dir)
    
    #birdEye_project(args.input_dir, args.output_dir, args.cfg_dir)