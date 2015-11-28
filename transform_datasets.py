# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python3.4/site-packages')
import cv2
import numpy as np
import pandas as pd

root_path='/home/mitya/Documents/CMake/mlschool/mlschool_01/'
train_data = pd.read_csv(root_path + 'final_train.csv', sep=',')
test_data = pd.read_csv(root_path + 'final_test.csv', sep=',')
train_data_folder = 'train_dataset/'
test_data_folder = 'test_dataset/'
arrays_folder = 'arrays/'

def transform_data(data, folder, limit=np.inf):
    sift = cv2.xfeatures2d.SIFT_create()
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + folder + str(row['image_id']) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts, descs = sift.detectAndCompute(gray, None)
        np.save('./' + arrays_folder + str(row['image_id']), descs)
        if index % 100 == 0:
            print(index, 'samples processed...')
            
transform_data(train_data, train_data_folder)
transform_data(test_data, test_data_folder)