# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import cluster, ensemble, cross_validation, metrics
from sklearn.externals import joblib

root_path='/home/mitya/Documents/CMake/mlschool/mlschool_01/'
train_data = pd.read_csv(root_path + 'final_train.csv', sep=',')
test_data = pd.read_csv(root_path + 'final_test.csv', sep=',')
train_data_folder = 'train_dataset/'
test_data_folder = 'test_dataset/'
arrays_folder = 'arrays/'

def get_random_subarray(arr, size=100):
    perm = np.random.permutation(min(size, arr.shape[0]))
    new_arr = None
    for index in range(perm.shape[0]):
        if index == 0:
            new_arr = arr[perm[index]]
        else:
            new_arr = np.vstack((new_arr, arr[perm[index]]))
    return new_arr
        
class bow:
    model = None; clusters=50; filename = './bag_of_words.txt'
    def __init__(self, n_clusters=50):
        self.model = joblib.load(self.filename)
        self.clusters = n_clusters
    def fit(self, data, folder, limit=100):        
        descriptors = None
        for index, row in data.iterrows():
            if index==limit:
                break
            descriptor = np.load('./' + arrays_folder + str(row['image_id']) + '.npy')
            descriptor = get_random_subarray(descriptor)
            if index == 0: 
                descriptors = descriptor
            else:                    
                descriptors = np.vstack((descriptors, descriptor))
            if index % 100 == 0:
                print(index, 'samples done...')
        print('clusterization...')
        model = cluster.KMeans(n_clusters=self.clusters, n_jobs=2)
        model.fit(descriptors)
        joblib.dump(model, self.filename)
        self.model = model
    def transform(self, data, folder, samples=np.inf):
        features = np.zeros((min(samples, data.shape[0]), self.clusters))
        for index, row in data.iterrows():
            if index==samples:
                break
            descriptor = np.load('./' + arrays_folder + str(row['image_id']) + '.npy')
            words = self.model.predict(descriptor)
            for word in words:
                features[index][word] += 1
            if index % 100 == 0:
                print(index, 'samples done...')
        return features

voc = bow(n_clusters = 50)
print('train bow vocabulary?')
answer = input()
if answer == 'y':
    voc.fit(train_data, train_data_folder, 1500)

print('reading data...')
X = voc.transform(train_data, train_data_folder)
y = train_data['image_label'].values[:X.shape[0]].ravel()

model = ensemble.RandomForestClassifier(n_estimators=240, criterion='entropy', max_features=0.99, n_jobs=2)
preds = model.fit(X, y).predict_proba(voc.transform(test_data, test_data_folder))[:, 1]
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)