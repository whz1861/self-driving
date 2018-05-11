#!/usr/bin/env python2.7
#coding=utf-8
# Baidu RPC - A framework to host and access services throughout Baidu.
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved

"""
lane detect
"""

import os
import sys
import cv2
import glob
import numpy as np
import argparse
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
sys.path.append(os.path.join(this_dir, '../'))


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def laneFit(edges, mask, templ_l, templ_m, templ_r):

    h, w = mask.shape[:2]
    
    edges_select = cv2.multiply(edges, mask/255)
    
    match_l = cv2.matchTemplate(mask, templ_l, cv2.TM_CCOEFF_NORMED)
    match_m = cv2.matchTemplate(mask, templ_m, cv2.TM_CCOEFF_NORMED)
    match_r = cv2.matchTemplate(mask, templ_r, cv2.TM_CCOEFF_NORMED)

    show_img = np.zeros(shape = (h, w, 3), dtype=np.uint8)
    apply_mask(show_img, mask/255, (0,1,0), 0.3)
    apply_mask(show_img, edges_select/255, (1,0,0), 0.5)
    
    _, con_l, _, loc_l = cv2.minMaxLoc(match_l)
    _, con_m, _, loc_m = cv2.minMaxLoc(match_m)
    _, con_r, _, loc_r = cv2.minMaxLoc(match_r)
    
    print loc_l
    #cv2.rectangle(show_img, loc_l, loc_l+templ_l.shape[:2], (0,0,255), 1)
    

    cv2.namedWindow('edges', 0)
    cv2.namedWindow('mask', 0)
    cv2.namedWindow('edges_s', 0)
    cv2.namedWindow('show_img', 0)
    cv2.imshow('edges', edges)
    cv2.imshow('mask', mask)
    cv2.imshow('edges_s', edges_select)
    cv2.imshow('show_img', show_img)
    cv2.waitKey(0)
    
    
    
    
    
    return


def laneFits(input_dir, output_dir):
    
    templ_l = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template0.jpg', 0)
    templ_m = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template1.jpg', 0)
    templ_r = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template2.jpg', 0)
    
    for fname in glob.glob(os.path.join(input_dir, "*edge.jpg")):
        edges = cv2.imread(fname, 0)
        
        mask_name = fname.replace('edge', 'mask')
        mask = cv2.imread(mask_name, 0)
        
        laneFit(edges, mask, templ_l, templ_m, templ_r)
    
    return


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='lane detection')
    parser.add_argument('--input_dir', dest='input_dir', help='',
                        default='./test', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='',
                        default='./test/result', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # set args
    args = parse_args()
    if args.input_dir == '':
        print(
            'Usage: python lane_detect.py '
            '               --input_dir dir_of_pic'
            '               --output_dir save_to_path')
        sys.exit(1)

    laneFits(args.input_dir, args.output_dir)


