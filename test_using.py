# -*- coding: utf-8 -*-
import pandas as pd
import bow

from sklearn import ensemble

def classification(model, get_features = True, train_bow = True):

    root_path = '/home/mitya/Documents/CMake/mlschool/mlschool_01/'
    csv_train = root_path + 'final_train.csv'
    csv_test = root_path + 'final_test.csv'
    features_folder = './arrays/'
    
    csv_reader = pd.read_csv(csv_train, sep = ',')

    vocabulary = bow.bow('bag_of_words', 'kmeans', 10)

    if get_features == True:
        vocabulary.transform_data(csv_train, root_path + 'train_dataset/', features_folder)
        vocabulary.transform_data(csv_test, root_path + 'test_dataset/', features_folder)

    if train_bow == True:
        vocabulary.fit(csv_train, features_folder, 100)

    X = vocabulary.transform(csv_train, features_folder, 20)
    y = csv_reader['image_label'].values[:X.shape[0]].ravel()

    preds = model.fit(X, y).predict_proba(vocabulary.transform(csv_test, features_folder, 10))[:, 1]
    #csv_reader = pd.read_csv(csv_test, sep = ',').drop('image_url', 1)
    #csv_reader['image_label'] = preds
    #csv_reader.to_csv(root_path + '/res.csv', index = False)

    return preds

model = ensemble.RandomForestClassifier(n_estimators = 20, criterion = 'entropy', max_features = 0.99, n_jobs = 2)

classification(model, False)