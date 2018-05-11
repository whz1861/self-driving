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
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse
import math
import copy

this_dir = os.path.dirname(os.path.abspath(__file__))
print('this_dir = {}'.format(this_dir))
sys.path.append(this_dir)
sys.path.append(os.path.join(this_dir, '..'))


from calibration.caliberation import undistort
from calibration.birdeye import birdEye_project
from lane_detect.lane_detect import lane_detect
from lane_detect.lane_detect import edgeDetects
from lane_detect.lane_detect import roadDetects
from lane_detect.lane_fit import laneFits

def run(input_dir, output_dir, cfg_dir):
    
    edge_path = os.path.join(output_dir, 'edges')
    print('deal edge..........')
    if not os.path.exists(edge_path):
        os.makedirs(edge_path)
        edgeDetects(input_dir, edge_path)
    
    undistort_path = os.path.join(output_dir, "undistort")
    print('deal undistort........')
    if not os.path.exists(undistort_path):
        os.makedirs(undistort_path)
        undistort(edge_path, undistort_path, cfg_dir)
    
    birdeye_path = os.path.join(output_dir, "birdege")
    print('deal birdege........')
    if not os.path.exists(birdeye_path):
        os.makedirs(birdeye_path)
        birdEye_project(undistort_path, birdeye_path, cfg_dir)

    
    

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
        
    run(args.input_dir, args.output_dir, args.cfg_dir)