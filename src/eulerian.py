"""
The program 'eulerian.py' reproduces the results of the paper Eulerian Video
Magnification for Revealing Subtle Changes in the World by Hao-Yu Wu, Michael
Rubinstein, Eugene Shih, John Guttag, Frédo Durand and William T. Freeman from
the MIT CSAIL and Quanta Research Cambridge, Inc (SIGGRAPH 2012).
"""

# Author: Baptiste Lefebvre <baptiste.lefebvre@ens.fr>
# License: MIT


import cv2 as cv
import math as mt
import numpy as np
import os
import scipy as sp
import scipy.misc



def fileparts(file):
    """
    Returns the directory, file name and file name extension for the specified
    file.
    """
    directory, base = os.path.split(file)
    name, extension = os.path.splitext(base)
    return (directory, name, extension)

def fullfile(directory, name, extension):
    """
    Builds a full file specification from the directory, file name and file name
    extension specified.
    """
    base = name + extension
    return os.path.join(directory, base)

def im2double(image):
    """
    Convert image to double precision.
    """
    return image.astype(np.float64)

def rgb2yiq(rgb_frame):
    """
    Change the color space of an image from RGB to YIQ.
    """
    shape = rgb_frame.shape
    M = np.array([[+0.299, +0.596, +0.211],
                  [+0.587, -0.274, -0.523],
                  [+0.114, -0.322, +0.312]])
    yiq_frame = np.zeros(shape, dtype=np.float64)
    for i in xrange(0, shape[0]):
        yiq_frame[i, :, :] = np.dot(rgb_frame[i, :, :], M)
    return yiq_frame

def blurDn(input_frame, level):
    """
    Blur and downsampling an image.
    """
    if 1 < level:
        output_frame = blurDn(cv.pyrDown(input_frame), level - 1)
    else:
        output_frame = cv.pyrDown(input_frame)
    return output_frame

def blurDnClr(input_frame, level):
    """
    Blur and downsampling a 3-color image.
    """
    temp = blurDn(input_frame[:, :, 0], level)
    shape = (temp.shape[0], temp.shape[1], input_frame.shape[2])
    output_frame = np.zeros(shape, dtype=np.float64)
    output_frame[:, :, 0] = temp
    for i in xrange(1, input_frame.shape[2]):
        output_frame[:, :, i] = blurDn(input_frame[:, :, i], level)
    return output_frame

def build_Gdown_stack(input_file, start_index, end_index, level):
    """
    Apply Gaussian pyramid decomposition on the input file from the start index
    to the end index and select a specific band indicated by level.
    """
    # Read video.
    input_video = cv.VideoCapture(input_file)
    # Extract video info.
    video_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_number_channels = 3
    _, rgb_frame = input_video.read()
    
    # First frame.
    rgb_frame = im2double(rgb_frame)
    yiq_frame = rgb2yiq(rgb_frame)
    
    blurred_image = blurDnClr(yiq_frame, level)
    
    # Create pyramidal stack.
    shape = (end_index - start_index + 1, blurred_image.shape[0],
             blurred_image.shape[1], blurred_image.shape[2])
    Gdown_stack = np.zeros(shape, dtype=np.float64)
    Gdown_stack[0, :, :, :] = blurred_image
    for k in xrange(start_index + 1, end_index + 1):
        _, rgb_frame = input_video.read()
        rgb_frame = im2double(rgb_frame)
        yiq_frame = rgb2yiq(rgb_frame)
        blurred_frame = blurDnClr(yiq_frame, level)
        Gdown_stack[k, :, :, :] = blurred_frame
    return Gdown_stack

def ideal_bandpassing(input, dimension, wl, wh, sampling_rate):
    """
    Apply ideal band pass filter on the input along the specified dimension.
    """
    input_shifted = np.rollaxis(input, dimension - 1)
    shape = input_shifted.shape
    
    n = shape[0]
    dn = len(shape)
    
    freq = np.arange(0, n, dtype=np.float)
    freq = freq / float(n) * float(sampling_rate)
    mask = np.logical_and(wl < freq, freq < wh)
    mask = np.reshape(mask, (mask.shape[0],) + (1,) * (len(shape) - 1))
    
    shape = (1,) + shape[1:];
    mask = np.tile(mask, shape)
    
    F = np.fft.fft(input_shifted, axis=0)
    F[np.logical_not(mask)] = 0
    output = np.fft.ifft(F, axis=0).real
    output = np.rollaxis(output, (dn - (dimension - 1)) % dn)
    return output

def imresize(frame, size):
    """
    Resize an image.
    """
    #return sp.misc.imresize(frame, size)
    return cv.resize(frame, size)

def yiq2rgb(yiq_frame):
    """
    Change the color space of an image from YIQ to RGB.
    """
    shape = yiq_frame.shape
    M = np.array([[+1.000, +1.000, +1.000],
                  [+0.956, -0.272, -1.106],
                  [+0.621, -0.647, +1.703]])
    rgb_frame = np.zeros(shape, dtype=np.float64)
    for i in xrange(0, shape[0]):
        rgb_frame[i, :, :] = np.dot(yiq_frame[i, :, :], M)
    return rgb_frame

def im2uint8(frame):
    """
    Convert image to 8-bit unsigned integers.
    """
    return frame.astype(np.uint8)

def amplify_spatial_Gdown_temporal_ideal(input_file, output_directory, alpha,
                                         level, fl, fh, sampling_rate,
                                         chrom_attenuation):
    """
    TODO: add description.
    """
    _, input_name, _ = fileparts(input_file)
    output_name = input_name \
                  + "-ideal-from-" + repr(fl) \
                  + "-to-" + repr(fh) \
                  + "-alpha-" + repr(alpha) \
                  + "-level-" + repr(level) \
                  + "-chromAtn-" + repr(chrom_attenuation)
    output_extension = ".avi"
    output_file = fullfile(output_directory, output_name, output_extension)
    
    # Read video.
    input_video = cv.VideoCapture(input_file)
    # Extract video info.
    video_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_number_channels = 3
    video_frame_rate = input_video.get(cv.CAP_PROP_FPS)
    if mt.isnan(video_frame_rate):
        video_frame_rate = 25.0
    video_length = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
    # Display video info.
    print("width: {0}" \
          "\nheight: {1}" \
          "\nnumber of channels: {2}" \
          "\nframe rate: {3}" \
          "\nlength: {4}".format(video_width,
                                 video_height,
                                 video_number_channels,
                                 video_frame_rate,
                                 video_length))
    temp = None
    
    start_index = 1 - 1
    end_index = video_length - 10 - 1; # TODO: understand why -10.
    
    video_fourcc = cv.VideoWriter_fourcc(*'FMP4')
    video_size = (video_width, video_height)
    output_video = cv.VideoWriter(output_file, video_fourcc, video_frame_rate, video_size)
    
    # Compute Gaussian blur stack.
    print("Spatial filtering...")
    Gdown_stack = build_Gdown_stack(input_file, start_index, end_index, level)
    print("Finished")
    
    # Temporal filtering.
    print("Temporal filtering...")
    filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, sampling_rate)
    print("Finished")

    # Amplify.
    filtered_stack[:, :, :, 0] = filtered_stack[:, :, :, 0] * alpha
    filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha * chrom_attenuation
    filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chrom_attenuation
    
    # Render on the input video.
    print("Rendering...")
    for k in xrange(start_index, end_index + 1):
        _, rgb_frame = input_video.read()
        rgb_frame = im2double(rgb_frame)
        yiq_frame = rgb2yiq(rgb_frame)

        filtered_frame = filtered_stack[k, :, :, :]
        filtered_frame = imresize(filtered_frame, (video_width, video_height))
        filtered_frame = yiq_frame + filtered_frame
        
        output_frame = yiq2rgb(filtered_frame)
        output_frame[output_frame < 0] = 0
        output_frame[255 < output_frame] = 255
        output_frame = im2uint8(output_frame)
        output_video.write(output_frame)
    print("Finished")
    
    # Release everything.
    input_video.release()
    output_video.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    #input_file = "./data/face.mp4"
    #input_file = "./data/face2.avi"
    #input_file = "./data/face3.avi"
    #input_file = "./data/face4.avi"
    input_file = "./data/face5.mp4"
    
    output_directory = "./data"
    
    print("Processing " + input_file)
    
    alpha = 50.0
    level = 3
    fl = 50.0 / 60.0
    fh = 60.0 / 60.0
    sampling_rate = 30
    chrom_attenuation = 1.0
    amplify_spatial_Gdown_temporal_ideal(input_file, output_directory, alpha,
                                         level, fl, fh, sampling_rate,
                                         chrom_attenuation)
