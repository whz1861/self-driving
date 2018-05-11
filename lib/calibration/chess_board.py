# coding = utf-8

import os
import cv2
import numpy as np

inch2mm = 25.4

def chess_board(board_size = (10, 7), dpi=600):
    a3_h, a3_w = (3508*2, 4961*2)
    
    w, h = board_size
    image = np.ones(shape=((h+2)*perBoardPixel, (w+2)*perBoardPixel, 1), dtype=np.uint8)*255
    
    for ind_v in xrange(1, w+1):
        for ind_h in xrange(1, h+1):
            flag = (ind_v + ind_h) % 2
            if flag == 0:
                left, right = ind_v*perBoardPixel, (ind_v+1)*perBoardPixel
                top, down = ind_h*perBoardPixel, (ind_h+1)*perBoardPixel
                #print left, right, top, down
                image[top:down, left:right, :] = 0
    print
    
    return image


if __name__ == '__main__':
    
    image = chess_board()
    cv2.imwrite('../data/chess_board.png', image)
    cv2.namedWindow('chess', 0)
    cv2.imshow('chess', image)
    cv2.waitKey(0)