from __future__ import unicode_literals
from primesense import openni2
from primesense import _openni2 as c_api

import cv2

import time
import sys
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage


# -------------------------------------------------------------
# input/output parameters

# input tensor
input_tensor_fn = 'tensors/black.pkl'

# output window size
FINAL_OUTPUT_WIDTH = 1440
FINAL_OUTPUT_HEIGHT = 900
# -------------------------------------------------------------


# -------------------------------------------------------------
# min/max depth parameters

# debug depth
show_depth = False
 
MIN_DEPTH = 500.0
MAX_DEPTH = 2200.0 

# keys "a" and "z" control the minimun depth cliping 
# keys "s" and "x" control the maximum depth cliping 

# this is how much the min/max values change at every click registered
KEY_DEPTH_OFFSET = 100
# -------------------------------------------------------------


global input_tensor
global INPUT_WIDTH
global INPUT_HEIGHT
global depth_levels
depth_levels = 256 

def load_tensor(input_tensor_fn):
    '''
    function that load a tensor created by either the video2input or the deepdream notebook
    '''
    global input_tensor
    global INPUT_WIDTH
    global INPUT_HEIGHT
    global depth_levels

    # input_tensor shape: (depth_leves, OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)
    input_tensor = np.load(input_tensor_fn)
    input_tensor = input_tensor[ np.linspace(0, input_tensor.shape[0]-1, num=depth_levels, endpoint=True, dtype=np.int32)[::-1] ,:,:,:]
    input_tensor = input_tensor[:,:,:, [2,1,0]]

    INPUT_WIDTH, INPUT_HEIGHT = input_tensor.shape[2], input_tensor.shape[1]
    print(' --------------------------------------')
    print 'Input shape:', input_tensor.shape, input_tensor.dtype
    print(' --------------------------------------')



def print_depth_frame_video(depth_data, input_tensor, thisType, show_depth):
    '''
    main function that does the merging of depth and input tensor
    '''
    # get depth in numpy and reshape
    img  = np.frombuffer(depth_data, dtype=thisType).astype(np.float32).reshape( (1, 424, 512) )

    # center and clip to MIN_DEPTH - MAX_DEPTH
    img[ np.where(img < MIN_DEPTH) ] = MAX_DEPTH+1000
    img -= MIN_DEPTH
    img = (np.clip(img, 0, MAX_DEPTH) / MAX_DEPTH) * (depth_levels-1)

    img = np.squeeze(img).astype(np.uint8)

    # optional: to remove artifacts, we use a dilation and gaussian filtering
    #  Feel free to comment the following two lines or play with the params
    img = ndimage.grey_dilation(img, footprint=np.ones((2,2)))
    img = ndimage.filters.gaussian_filter(img, sigma=0.3)
    
    # resize to input_tensor/input size
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))#, interpolation=cv2.INTER_LINEAR)
    
    # for debugging
    if show_depth:
        return img

    # python magic
    i, j = img.shape
    i, j = np.ogrid[:i, :j]
    image = input_tensor[img, i, j, :]

    if FINAL_OUTPUT_WIDTH > INPUT_WIDTH:
        image = cv2.resize(image, (FINAL_OUTPUT_WIDTH, FINAL_OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return image


# load the input
load_tensor(input_tensor_fn)

openni2.initialize()    
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 512, resolutionY = 424, fps = 60))

cv2.namedWindow("Slihouettes")          

# vars to count fps
start = time.time()
count_frames = 0

depth_stream.start()
while 1:
    frame = depth_stream.read_frame()
    depth_data = frame.get_buffer_as_uint16()

    video_image = print_depth_frame_video(depth_data, input_tensor, np.uint16, show_depth)
    cv2.imshow("Slihouettes", video_image)
    
    # silly fps counting
    count_frames += 1
    fps  = count_frames / (time.time() - start)
    if count_frames % 10 == 0:
        print( 'fps:', fps)
    
    # key bindings
    # keys "a" and "z" control the minimun depth cliping 
    # keys "s" and "x" control the maximum depth cliping 

    k = cv2.waitKey(5) & 0xFF
    if k == 27: # ESC TO EXIT
        break
    k = cv2.waitKey(33) & 0xFF
    if k == ord('a'):
        MIN_DEPTH += KEY_DEPTH_OFFSET
        print(' ------------------- MIN_DEPTH:', MIN_DEPTH)
    if k == ord('z'):
        MIN_DEPTH -= KEY_DEPTH_OFFSET
        print(' ------------------- MIN_DEPTH:', MIN_DEPTH)
    if k == ord('s'):
        MAX_DEPTH -= KEY_DEPTH_OFFSET
        print(' ------------------- MAX_DEPTH:', MAX_DEPTH)
    if k == ord('x'):
        MAX_DEPTH += KEY_DEPTH_OFFSET
        print(' ------------------- MAX_DEPTH:', MAX_DEPTH)

cv2.destroyAllWindows()
depth_stream.stop() 