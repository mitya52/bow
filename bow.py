# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/lib/python3.4/site-packages')

import cv2
import random
import numpy as np
import pandas as pd
#import pandas as pd

from sklearn import cluster
from sklearn.externals import joblib

class bow:

    def __init__(self, bow_filename = 'bag_of_words', model = 'kmeans', n_clusters = 50):

        self.vocabulary = joblib.load(bow_filename)
        self.bow_filename = bow_filename
        self.clusters = n_clusters
        self.model = model

    def fit(self, csv_filename, features_folder, limit_samples = np.inf, n_feature_from_sample = 10, print_log = False):

        descriptors = None
        
        csv_reader = pd.read_csv(csv_filename, sep = ',')

        for index, row in csv_reader.iterrows():

            if index == limit_samples:
                break

            descriptor = np.load(features_folder + str(row['image_id']) + '.npy')
            random.shuffle(descriptor)
            descriptor = descriptor[:n_feature_from_sample,:]

            if index == 0:
                descriptors = descriptor
            else:                    
                descriptors = np.vstack((descriptors, descriptor))
            
            if print_log and index % 1000 == 0: print(index, 'samples done...')

        if print_log == True: 
            print('clusterization...')
        
        if self.model == 'kmeans':
            self.vocabulary = cluster.KMeans(n_clusters = self.clusters, n_jobs = 2)
        else:
            return
        
        self.vocabulary.fit(descriptors)
        
        joblib.dump(self.vocabulary, self.bow_filename)

    def transform(self, csv_filename, features_folder, limit_samples = np.inf, print_log = False):

        csv_reader = pd.read_csv(csv_filename, sep = ',')

        features = np.zeros((min(limit_samples, csv_reader.shape[0]), self.clusters))

        for index, row in csv_reader.iterrows():

            if index == limit_samples:
                break

            descriptor = np.load(features_folder + str(row['image_id']) + '.npy')
            words = self.vocabulary.predict(descriptor)

            for word in words:
                features[index][word] += 1

            if print_log and index % 1000 == 0: print(index, 'samples done...')

        return features

    def fit_transform(self, csv_filename, features_folder, limit_samples = np.inf, n_feature_from_sample = 10, print_log = False):

        self.fit(csv_filename, features_folder, limit_samples, n_feature_from_sample, print_log)

        return self.transform(csv_filename, features_folder, limit_samples, print_log)
    
    def transform_data(csv_filename, samples_folder, features_folder, limit_samples = np.inf, print_log = False):

        csv_reader = pd.read_csv(csv_filename, sep = ',')

        sift = cv2.xfeatures2d.SIFT_create()
        
        for index, row in csv_reader.iterrows():
    
            if index == limit_samples:
                break
    
            image = cv2.imread(samples_folder + str(row['image_id']) + '.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
            kpts, descs = sift.detectAndCompute(gray, None)
            np.save(features_folder + str(row['image_id']), descs)
    
            if print_log and index % 100 == 0: print(index, 'samples processed...')
        
        return samples_folder