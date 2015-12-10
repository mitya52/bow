# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python3.4/site-packages')

import cv2
import numpy as np
import pandas as pd

def transform_data(csv_reader, samples_folder, features_folder, limit_samples = np.inf, print_log = False):

    sift = cv2.xfeatures2d.SIFT_create()
    
    for index, row in csv_reader.iterrows():

        if index == limit_samples:
            break

        image = cv2.imread(samples_folder + str(row['image_id']) + '.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kpts, descs = sift.detectAndCompute(gray, None)
        np.save(features_folder + str(row['image_id']), descs)

        if print_log and index % 100 == 0: print(index, 'samples processed...')