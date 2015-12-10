# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd

from sklearn import cluster, ensemble
from sklearn.externals import joblib

class bow:

    def __init__(self, filename, n_clusters = 50):

        if filename != None:
            self.model = joblib.load(filename)

        self.filename = filename
        self.clusters = n_clusters

    def fit(self, csv_reader, features_folder, limit_samples = np.inf, n_feature_from_sample = 10, n_jobs = 2, print_log = False):

        descriptors = None

        for index, row in csv_reader.iterrows():

            if index == limit_samples:
                break

            descriptor = np.load(features_folder + str(row['image_id']) + '.npy')
            descriptor = random.shuffle(descriptor)[: n_feature_from_sample, :]            


            if index == 0:
                descriptors = descriptor
            else:                    
                descriptors = np.vstack((descriptors, descriptor))
            
            if print_log and index % 1000 == 0: print(index, 'samples done...')
                
        if print_log: print('clusterization...')
        
        model = cluster.KMeans(n_clusters = self.clusters, n_jobs = n_jobs)

        model.fit(descriptors)
        joblib.dump(model, self.filename)

        self.vocabulary = model

    def transform(self, csv_reader, features_folder, limit_samples = np.inf, print_log = False):

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

    def fit_transform(self, csv_reader, features_folder, limit_samples = np.inf, n_feature_from_sample = 10, n_jobs = 2, print_log = False):

        self.fit(csv_reader, features_folder, limit_samples, n_feature_from_sample, n_jobs, print_log)

        return self.transform(csv_reader, features_folder, limit_samples, print_log)