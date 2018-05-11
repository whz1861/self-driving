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



#birdEye
M = np.array([[-1.13003126e+00, -2.02812814e+00, 3.06759898e+02],
              [-1.85580377e-01, -4.29373592e+00, 4.96542491e+02],
              [-6.37672254e-04, -1.34383082e-02, 1.00000000e+00]], dtype=np.float32)
# load calib file
calib_path = '/home/leo/disk/project/baidu/code/hackthon/lib/calibration/result'
mapx = np.load(os.path.join(calib_path, 'mapx.npy'))
mapy = np.load(os.path.join(calib_path, 'mapy.npy'))
new_nk = np.load(os.path.join(calib_path, 'new_nk.npy'))
dist_coefs = np.load(os.path.join(calib_path, 'dist_coefs.npy'))

birdeye_p = np.load(os.path.join(calib_path, 'birdEyeProject.npy'))
birdeye_p_inv = np.load(os.path.join(calib_path, 'birdEyeProject_inv.npy'))

templ_l = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template0.jpg', 0)
templ_m = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template1.jpg', 0)
templ_r = cv2.imread('/home/leo/disk/project/baidu/code/hackthon/cfg/template2.jpg', 0)


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def edgeDetect(image, show=False):
    # cvt gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_copy = copy.deepcopy(gray)

    # denoise
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # edge detect
    low_threshold = 50
    high_threshold = 150
    edge = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # binary
    thresh, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    
    #road detect
    mask = roadDetect(bin_img, show=False)

    #edge by mask
    edge_mask = cv2.multiply(mask / 255, edge)

    laneFit(image, edge, mask, mapx, mapy, birdeye_p)

    show_img = copy.deepcopy(image)
    apply_mask(show_img, mask / 255, (0, 1, 0), alpha=0.3)
    apply_mask(show_img, edge / 255, (0, 0, 1), alpha=0.5)
    apply_mask(show_img, edge_mask / 255, (1, 0, 0), alpha=0.5)
    
    if show:
        cv2.namedWindow("image", 0)
        cv2.namedWindow("gray", 0)
        cv2.namedWindow("edge", 0)
        cv2.namedWindow('bin_img', 0)
        cv2.imshow('image', image)
        cv2.imshow('gray', gray)
        cv2.imshow('edge', edge)
        cv2.imshow('bin_img', bin_img)
        cv2.namedWindow("show_img", 0)
        cv2.imshow("show_img", show_img)
        
        cv2.waitKey(0)
        
    return gray, edge, bin_img, mask, show_img

def edgeDetects(input_dir, output_dir):
    
    #for fname in glob.glob(os.path.join(input_dir, "*.jpg")):
    for index in xrange(1,500):
        fname = os.path.join(input_dir, 'test{}.jpg'.format(index))
        print('~~~~~~~~~~~~edge:{}~~~~~~~~~~~~~~'.format(fname))
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray, edge, bin_img, road_mask, show_img = edgeDetect(img, show=False)
        
        save_name = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_show.jpg'))
        cv2.imwrite(save_name, show_img)
        save_name = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_edge.jpg'))
        cv2.imwrite(save_name, edge)
        save_name = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_mask.jpg'))
        cv2.imwrite(save_name, road_mask)
        save_name = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_src.jpg'))
        cv2.imwrite(save_name, img)
        save_name = os.path.join(output_dir, os.path.basename(fname).replace('.jpg', '_bin.jpg'))
        cv2.imwrite(save_name, bin_img)
        
def roadDetect(bin_img, show=False):
    
    #open
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, element)

    #erode
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
    erode = cv2.erode(open, element)

    # find contours
    dst, contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    loc = -1
    max = 0
    for index, con in enumerate(contours):
        area = cv2.contourArea(con)
        if area > max:
            max = area
            loc = index

    mask = np.zeros(shape=(erode.shape[0], erode.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, contours, loc, 255, -1)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    mask = cv2.dilate(mask, element)
    
    """
    # fit mask
    points = np.asarray(contours[loc], dtype=np.float32)
    points = np.squeeze(points)
    print('points = {}'.format(points.shape))
    z = np.polyfit(points[:,0], points[:,1], 2)
    print('polyfit = {}'.format(z))
    """
    
    if show:
        """
        h,w = mask.shape[:2]
        mask_img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask_img[:,:,0] = np.array(mask, dtype=np.uint8)
        p2 = np.poly1d(z)
        for i in xrange(0, mask.shape[1]):
            x = i
            y = p2(x)
            cv2.rectangle(mask_img, (int(x),int(y)), (int(x+1), int(y+1)), (0,0,255), -1)
        """
        cv2.namedWindow('image', 0)
        cv2.namedWindow('erode', 0)
        cv2.namedWindow('contours', 0)
        cv2.imshow('image', bin_img)
        cv2.imshow('erode', erode)
        cv2.imshow('contours', mask)
        
        cv2.waitKey(0)
    
    return mask
    
def roadDetects(input_dir, output_dir):
    
    for fname in glob.glob(os.path.join(input_dir, "*bin.jpg")):
        print('~~~~~~~~~~~~~~~~{}~~~~~~~~~~~~~~~~'.format(fname))
        img = cv2.imread(fname, 0)
        print img.shape
        if img is None:
            continue
    
        roadDetect(img, show=False)


def laneFit(image, edge, mask, mapx, mapy, birdeye_p):
    
    h, w = edge.shape[:2]
    edge = cv2.multiply(edge, mask/255)
    
    # find contours
    dst, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_con = np.zeros(shape=(edge.shape[0], edge.shape[1]), dtype=np.uint8)
    
    show_img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    apply_mask(show_img, mask/255, (1,0,0), 0.5)
    
    
    # undistort and birdeye
        
    image_u = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    image_b = cv2.warpPerspective(image_u, birdeye_p, (w, h))
    edge_u = cv2.remap(edge, mapx, mapy, cv2.INTER_LINEAR)
    edge_b = cv2.warpPerspective(edge_u, birdeye_p, (w, h))
    mask_u = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)
    mask_b = cv2.warpPerspective(mask_u, birdeye_p, (w, h))
    
    
    edge_n = np.zeros_like(edge_b, dtype=np.uint8)
    edge_n[:, 240:400] = edge_b[:, 240:400]
    # fit line by Hough
    # line detect by HoughLinesP
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edge_n, rho, theta, threshold, np.array([]), min_line_length,
                            max_line_gap)
    
    threshVal, mask_bb = cv2.threshold(mask_b, 127, 255, cv2.THRESH_BINARY)
    apply_mask(image_b, mask_bb/255, (0,1,0))
    cv2.line(image_b, (w/2, 0), (w/2, h), (0,0,255), 2)
    
    infor_top = []
    infor_bottom = []
    lines = np.squeeze(lines, axis=1)
    for line in lines:
        cv2.line(image_b, (line[0], line[1]), (line[2], line[3]), (255,0,0), 1)
        
        # calc the top joint point
        join_x = w/2
        if line[2] == line[0]:
            join_y = 0
        else:
            join_y = (join_x - line[0]) * (line[3] - line[1]) / (line[2] - line[0]) + line[1]
        
        if join_x > np.minimum(line[0], line[2]) - 5 and \
            join_x < np.maximum(line[0], line[2]) + 5 and \
            join_y > np.minimum(line[1], line[3]) - 5 and \
            join_y < np.maximum(line[1], line[3]) + 5:
            #cv2.rectangle(image_b, (join_x, int(join_y)), (join_x+1, int(join_y+1)), (0,255,255), 1)
            if line[2] == line[0]:
                angle = 0
            else:
                angle = math.atan(float(line[3]-line[1])/float(line[2]-line[0])) / math.pi * 180
                if angle > 0:
                    angle = -(90 - angle)
                else:
                    angle = 90 + angle
                
            infor_top.append([join_y, angle])

        # calc infor bottom
        join_y = 440
        if line[3] == line[1]:
            join_x = 0
        else:
            join_x = (join_y - line[1]) * (line[2] - line[0]) / (line[3] - line[1]) + line[0]
        
        if join_x > 200 and join_x < 440:
            infor_bottom.append(join_x)
            cv2.circle(image_b, (int(join_x), int(join_y)), 3, (0,255,255), -1)

    
            #print('angle = {}'.format(angle))
            
            #cv2.namedWindow('image_b', 0)
            #cv2.imshow('image_b', image_b)
            #cv2.waitKey(0)
    if len(infor_top) != 0:
        infor = sorted(infor_top, key=lambda x:x[0])
        top_p = np.mean(np.array(infor[:5], dtype=np.float32), axis=0)
        print('infor = {}'.format(infor))
        print('top_p = {}'.format(top_p))
        cv2.circle(image_b, (w/2, int(top_p[0])), 3, (0,255,255), -1)



    cv2.namedWindow('image_b', 0)
    cv2.namedWindow('edge_b', 0)
    cv2.namedWindow('mask_b', 0)
    cv2.namedWindow('edge_n', 0)
    cv2.namedWindow('mask', 0)
    cv2.imshow('image_b', image_b)
    cv2.imshow('edge_b', edge_b)
    cv2.imshow('mask_b', mask_b)
    cv2.imshow('edge_n', edge_n)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    
    
    
    
    return
    weight = np.arange(0, w)
    print weight
    for i in xrange(0, h):
        mid = np.sum(weight * mask[i, :]/255) / (np.sum(mask[i,:])/255 + 1)
        x, y = mid, i
        cv2.rectangle(show_img, (int(x), int(y)), (int(x + 1), int(y + 1)), (0, 0, 255), 1)
        
    # show contours by point
    for con in contours:
        points = np.squeeze(con, axis=1)
        for pt in points:
            x, y = pt
            cv2.rectangle(show_img, (int(x), int(y)), (int(x + 1), int(y + 1)), (0, 0, 255), 1)
    
            cv2.namedWindow('con', 0)
            cv2.imshow("con", edge_con)
            cv2.namedWindow('show', 0)
            cv2.imshow('show', show_img)
            cv2.waitKey(0)
        
    
    """
    for i in xrange(0, len(contours)):
        cv2.drawContours(edge_con, contours, i, 255, -1)
        points = np.asarray(contours[i], dtype=np.float32)
        points = np.squeeze(points)
        if len(points) < 5:
            continue
        print('{}: len points:{}'.format(i, len(points)))
        z1 = np.polyfit(points[:, 0], points[:, 1], 1)
        p1 = np.poly1d(z1)
        z2 = np.polyfit(points[:, 0], points[:, 1], 1)
        p2 = np.poly1d(z2)
        
        mse1 = np.sum((p1(points[:,0])-points[:,1])**2)/len(points)
        mse2 = np.sum((p2(points[:,0])-points[:,1])**2)/len(points)
        
        if mse1 < mse2:
            p = p1
        else:
            p = p2

        for x in xrange(np.min(points[:, 0]), np.max(points[:,0])):
            y = p(x)
            cv2.rectangle(show_img, (int(x), int(y)), (int(x+1), int(y+1)), (0,0,255), 1)
        
        cv2.namedWindow('con', 0)
        cv2.imshow("con", edge_con)
        cv2.namedWindow('show', 0)
        cv2.imshow('show', show_img)
        cv2.waitKey(0)
    """
    return




def lane_detect(image, save_name=None):
    start = cv2.getTickCount()
    src_image = copy.deepcopy(image)
    h, w, c = image.shape
    
    #cvt gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_copy = copy.deepcopy(gray)
    # denoise
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    print gray.shape
    sub_img = gray[200:,150:200]
    print np.mean(sub_img)
    thresh, bin_image = cv2.threshold(blur_gray, np.mean(sub_img), 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    
    # edge detect
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(50,imshape[0]),(420, 280), (550, 280), (950,imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # line detect by HoughLinesP
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    # select lines
    lines = np.squeeze(lines)
    s_lines = []
    for line in lines:
        slope = float(line[3] - line[1]) / float(line[2] - line[0] + 0.00001)
        rho = math.atan(slope)
        rho_d = rho / np.pi * 180
        
        line_center = ((line[0]+line[2])/2.0, (line[1]+line[3])/2,0)
        
        dist1 = np.sqrt((line[0] - 160) * (line[0] - 160) +
                           (line[1] - 240) * (line[1] - 240))
        dist2 = np.sqrt((line[2] - 160) * (line[2] - 160) +
                           (line[3] - 240) * (line[3] - 240))
        
        if rho_d > -30 and rho_d < 30:
            continue
        elif line_center[1] < h * 0.33:
            continue
        elif np.minimum(dist1, dist2) > 160:
            continue
        else:
            s_lines.append(line)
            
    print('len liens = {}'.format(len(lines)))
    print('len s_lines = {}'.format(len(s_lines)))
    
    end = cv2.getTickCount()
    print('exeTime = {}'.format((end-start)*1000/cv2.getTickFrequency()))
    
    for line in s_lines:
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0,0,255), 1)

    color_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.line(color_gray, (line[0], line[1]), (line[2], line[3]), (255,0,0), 1)

    if save_name is not None:
        save_file = save_name.replace('.jpg', '_show.jpg')
        cv2.imwrite(save_file, image)
        save_file = save_name.replace('.jpg', '_edges.jpg')
        cv2.imwrite(save_file, edges)
        save_file = save_name.replace('.jpg', '_bin.jpg')
        cv2.imwrite(save_file, bin_image)
        save_file = save_name.replace('.jpg', '_gray.jpg')
        cv2.imwrite(save_file, color_gray)
    else:
        cv2.namedWindow('show', 0)
        cv2.imshow('show', image)
        cv2.namedWindow('edges', 0)
        cv2.imshow('edges', edges)
        cv2.namedWindow('bin', 0)
        cv2.imshow('bin', bin_image)
        cv2.namedWindow('gray', 0)
        cv2.imshow('gray', color_gray)

        undistort_edge = cv2.remap(edges, mapx, mapy, cv2.INTER_NEAREST)
        bird_edge = cv2.warpPerspective(undistort_edge, M, (w, h))
        cv2.namedWindow('bird_edge', 0)
        cv2.imshow('bird_edge', bird_edge)
        
        cv2.waitKey(0)
    
    return


def road_detect(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    h, w= mask.shape
    mask[int(h/2):int(h), int(w/4):int(w*0.75)] = 1
    gray_mask = cv2.multiply(gray, mask)
    
    thresh = np.mean(gray[int(0):int(h/2), 0:w])
    print thresh
    thresh, bin_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    print('bin_img.shape = {}'.format(bin_img.shape))
    
    cv2.namedWindow('gray', 0)
    cv2.namedWindow('mask', 0)
    cv2.namedWindow('gray_mask', 0)
    cv2.namedWindow('bin_img', 0)
    cv2.imshow('gray', gray)
    cv2.imshow('mask', mask)
    cv2.imshow('gray_mask', gray_mask)
    cv2.imshow('bin_img', bin_img)
    cv2.waitKey(0)
    
    return
    
    h, w, c = image.shape
    mb = np.mean(image[200:230, 140:180, 0])
    mg = np.mean(image[200:230, 140:180, 1])
    mr = np.mean(image[200:230, 140:180, 2])
    print("mean = {}, {}, {}".format(mb, mg, mr))
    
    inrange = cv2.inRange(image, (mb-30, mg-30, mr-30), (mb+30, mg+30, mr+30))
    print inrange.shape
    cv2.namedWindow('img', 0)
    cv2.imshow('img', inrange)
    cv2.waitKey(0)
    
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
        
    print math.atan(-1)
    print math.atan(-0.05)
    
    

    
    
    for index in xrange(0,299):
        i = index % 299 + 1
        pic_name = os.path.join(args.input_dir, "{}.jpg".format(i))
        print pic_name
        image = cv2.imread(pic_name)

        undistort_image = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)
        print('undistort_image.shape = {}'.format(undistort_image.shape))
        h, w, c = undistort_image.shape
        bird_image = cv2.warpPerspective(undistort_image, M, (w, h))
        print('bird_image.shape = {}'.format(bird_image.shape))
        #road_detect(bird_image)
        
        #continue
        
        
        cv2.namedWindow('image', 0)
        cv2.namedWindow('undistort', 0)
        cv2.namedWindow('bird', 0)
        cv2.imshow('image', image)
        cv2.imshow('undistort', undistort_image)
        cv2.imshow('bird', bird_image)
        cv2.waitKey()
        
        
        #continue
        save_name = os.path.join(args.output_dir, "{}.jpg".format(i))
        #lane_detect(image, save_name)
        lane_detect(image)

        #fig.savefig(save_name)
    
    
