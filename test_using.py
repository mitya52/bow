# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import feature_getter as fg
import bow as fe

from sklearn import ensemble

def classification(model, get_features = True, train_bow = True):

    root_path = '/home/mitya/Documents/CMake/mlschool/mlschool_01/'
    csv_reader_train = pd.read_csv(root_path + 'final_train.csv', sep=',')
    csv_reader_test = pd.read_csv(root_path + 'final_test.csv', sep=',')
    train_data_folder = root_path + 'train_dataset/'
    test_data_folder = root_path + 'test_dataset/'
    features_folder = './arrays/'

    if get_features:
        fg.transform_data(csv_reader_train, train_data_folder, features_folder)
        fg.transform_data(csv_reader_test, test_data_folder, features_folder)

    vocabulary = fe.bow('bag_of_words', 10)

    if train_bow:
        vocabulary.fit(csv_reader_test, features_folder, 1000)

    X = vocabulary.transform(csv_reader_train, train_data_folder)
    y = csv_reader_train['image_label'].values[:X.shape[0]].ravel()

    preds = model.fit(X, y).predict_proba(vocabulary.transform(csv_reader_test, test_data_folder))[:, 1]
    csv_reader_test = csv_reader_test.drop('image_url', 1)
    csv_reader_test['image_label'] = preds
    csv_reader_test.to_csv(root_path + '/res.csv', index = False)

    return preds


model = ensemble.RandomForestClassifier(n_estimators = 240, criterion = 'entropy', max_features = 0.99, n_jobs = 2)

classification(model, False)