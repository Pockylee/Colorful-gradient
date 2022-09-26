#!/usr/bin/env python
# coding: utf-8
# Last modification: PO-YI, LI (20220926)

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import cv2
import numpy as np
import shutil
import math
import argparse
import multiprocessing
from threading import Thread
from multiprocessing import Process, Pool
from operator import add, sub, mul
from sklearn import preprocessing
from imutils import paths


def image_colorfulness(image):
    #Split the image into R,G,B (Notice: R,G,B are all vectors)
    (B, G, R) = cv2.split(image.astype("float"))
    #rg = R - G
    rg = np.absolute(R - G)
    #yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    #Caculate the standard deviation and mean of rg, yb
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    #Caculate the standard deviation and mean of rgyb
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    #Get the final answer of colorfulness
    return stdRoot + (0.3 * meanRoot)

def image_sharpness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gy, gx = np.gradient(image)
    gnorm = np.sqrt(gx**2 + gy**2)
    return np.average(gnorm)

def image_cast(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    h,w,_ = image.shape
    da = a_channel.sum()/(h*w)-128
    db = b_channel.sum()/(h*w)-128
    histA = [0]*256
    histB = [0]*256
    for i in range(h):
        for j in range(w):
            ta = a_channel[i][j]
            tb = b_channel[i][j]
            histA[ta] += 1
            histB[tb] += 1
    msqA = 0
    msqB = 0
    for y in range(256):
        msqA += float(abs(y-128-da))*histA[y]/(w*h)
        msqB += float(abs(y - 128 - db)) * histB[y] / (w * h)
    result = math.sqrt(da*da+db*db)/math.sqrt(msqA*msqA+msqB*msqB)
    return result

def list_minmaxscaler(target_value_list, scaled_range = (0,10)):
    target_value_list = np.array(target_value_list)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=scaled_range)
    target_value_list = min_max_scaler.fit_transform(target_value_list.reshape(-1,1))
    target_value_list = target_value_list.reshape(1,-1)[0]

    return target_value_list

def single_case(image_path, output_path, parallel):
    image_file_sorted = sorted(paths.list_images(image_path))
    count = 0
    Colorfulness_list = []
    Sharpness_list = []
    Color_cast_list = []

    for index, image_file in enumerate(image_file_sorted):
        print("Current progress: ", count, "/", len(image_file_sorted),end="\r")
        count += 1
        image = cv2.imread(image_file)             # Read image from directory with opencv
        
        if parallel == False:
            color_cast_score = image_cast(image)       # Image preprocessing with color correction    
            sharpness = image_sharpness(image)         # Sharpness calculation
            colorfulness = image_colorfulness(image)   # Colorfulness calculation

        else:
            thread_color_cast = Thread(target=image_cast, args=[image])
            thread_sharpness = Thread(target=image_sharpness, args=[image])
            thread_colorfulness= Thread(target=image_colorfulness, args=[image])

            thread_color_cast.start()
            thread_sharpness.start()
            thread_colorfulness.start()

            sharpness = thread_sharpness.join()
            colorfulness = thread_colorfulness.join()
            color_cast_score = thread_color_cast.join()
        
        # Store both values into list.
        Colorfulness_list.append(colorfulness)
        Sharpness_list.append(sharpness)
        Color_cast_list.append(color_cast_score)

    # Min max scaler for Sharpness and Colorfulness list
    Sharpness_list = list_minmaxscaler(Sharpness_list)
    Colorfulness_list = list_minmaxscaler(Colorfulness_list)
    Color_cast_list = list_minmaxscaler(Color_cast_list, scaled_range = (1,10))
    
    #caculate the color-gradient function
    Color_cast_list = 1/Color_cast_list            # Reciprocal number of each value in color_cast_list 
    Color_score_list = list(map(mul, Color_cast_list, Colorfulness_list))
    Summary_list = np.array(list(map(add, Sharpness_list, Color_score_list)))
        
    index_max = np.argmax(Summary_list)
    print("                             ", end="\r")

    # Save_file
    image_dir = os.path.basename(image_path)
    if not os.path.exists(os.path.join(output_path,image_dir)):
        os.mkdir(os.path.join(output_path,image_dir))
    if not os.path.exists(os.path.join(output_path,image_dir, os.path.basename(image_file_sorted[index_max]))):
        shutil.copyfile(image_file_sorted[index_max], os.path.join(output_path,image_dir, os.path.basename(image_file_sorted[index_max])))
    
    return index_max, os.path.join(output_path,image_dir, os.path.basename(image_file_sorted[index_max]))

def multiple_case(image_dir_path, output_path, parallel):
    for image_dir in sorted(os.listdir(image_dir_path)):
        if image_dir.startswith("."): continue      # Avoid opening hidden directories.
        image_dir = os.path.join(image_dir_path,image_dir)
        print("Input Path: ", image_dir)
        max_index, max_out_path = single_case(image_dir, output_path, parallel)
        print("Explicit Index: ", max_index)
        print("Output Path: ", max_out_path)
        print("===================================================")

if __name__ == "__main__":
    #Argument customize in this section
    parser = argparse.ArgumentParser(description='An Microscope Image Auto-Focus Method based on Colorful-Gradient')
    parser.add_argument('-t', "--case_type", type=str, default="Multiple", help='Type of cases in the input image directory.(single/multiple)')
    parser.add_argument('-i', "--dir_path", type=str, default="./images/multiple", help='Please enter your image directory path here.')
    parser.add_argument('-o', "--output_path", type=str, default="./output", help="Please enter the output path where you want to store your explicit image in each cases")
    parser.add_argument('-p', "--parallel_compute", action='store_false', help="Choose to calculate the colorful-gradient in parallel computing mode or not.(True/False)")
    args = parser.parse_args()

    image_path = args.dir_path
    case_type = args.case_type
    output_path = args.output_path
    parallel_compute = args.parallel_compute

    if not os.path.exists(output_path): os.mkdir(output_path)  # Check outpu directory exist, create one if not.

    print("Program start")
    print("Task Type: ", case_type)
    print("Parallel Compute: ", parallel_compute)
    print("===================================================")
    if case_type == "single" or case_type == "Single":
        explicit_index, explicit_output_path = single_case(image_path, output_path, parallel_compute)
        print("Input Path: ", image_path)
        print("Explicit Index: ", explicit_index)
        print("Output Path: ", explicit_output_path)
    elif case_type == "multiple" or case_type == "Multiple":
        multiple_case(image_path, output_path, parallel_compute)
    else:
        raise Exception("You can only type in 'single' or 'multiple' in case_type argument!")

    print("End of the program. See you next time!")
