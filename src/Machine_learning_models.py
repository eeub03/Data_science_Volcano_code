import os
from glob import glob
import random
import numpy as np
import pandas as pd
import dask.dataframe as dd


os.chdir("C:/Users/Joe/PycharmProjects/Data_science_Volcano_code")

# Import the classifier models.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
import keras
file_save = open("accuracy", 'w')
PATH = "data\\train"
EXT = "*.csv"

all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

all_csv_files.sort(key=len, reverse=False)


# Sample used for feature selection
training_sample_size = 100
file_size = 100
training_sample_files = all_csv_files[0:file_size]
c = ["segment_id","sensor_1","sensor_2","sensor_3","sensor_4","sensor_5","sensor_6","sensor_7","sensor_8","sensor_9","sensor_10",
     ]
data = np.array([[np.zeros([training_sample_size])]*11]*len(training_sample_files))
i = 0


labels = pd.read_csv("data\\train.csv")
labels['time_to_eruption'] = labels['time_to_eruption'].round(decimals=-2)
labels = labels.sort_values(by=['segment_id'], axis=0)
labels = labels.to_numpy()
test_label = all_csv_files[0].split('\\')[2].split('.')[0]
print(labels.shape)
index_label = np.where(labels[:,0] == int(test_label))


sample_labels = labels[0:training_sample_size]
sample_labels_2 = sample_labels[:,1]

sample_labels_2 = ((((sample_labels_2 / 60) / 60) / 24)/ 7).round(decimals = 0)


sample_labels_2[sample_labels_2 > 74] = 75
counts, amount = np.unique(sample_labels_2, return_counts=True)

labels = sample_labels_2
try:
    f = open("data.csv")
    f.close()
except:
    for file in training_sample_files:
        print(i)
        pd_data = dd.read_csv(file, engine='python')
        pd_data = pd_data.compute()

        pd_data = pd_data.sample(n=training_sample_size, replace=True)
        pd_data.insert(10, "Label", [labels[i]] * training_sample_size)
        pd_data = pd_data.to_numpy()

        for j in range(0, 11):
            data[i, j] = pd_data[:, j]

        i += 1
    print(data.shape)
    data = np.transpose(data, [2, 0, 1])
    print(data.shape)
    data = data.reshape(training_sample_size * len(training_sample_files), 11)
    pd.DataFrame(data).to_csv("data.csv")


scoring = {'accuracy': make_scorer(accuracy_score)}
data = pd.read_csv("data.csv")
data = data.replace(np.nan, 0)
print(data.shape)
data = data.to_numpy()


# Smote
labels = data[:,11]
df_data = data[:,1:11]
# Feature selection

over = SMOTE()


clf = DecisionTreeClassifier()
# 5 times cross val
cv_environment = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3)

scoring = {'accuracy': make_scorer(accuracy_score),
           'f1_score': make_scorer(f1_score, average='macro')}


new_data = []
try:
    new_data = pd.read_csv("C:\\Users\\Joe\\PycharmProjects\\Data_science_Volcano_code\\data_sfs.csv")
    new_data = new_data.to_numpy()
except:
    sfs = SequentialFeatureSelector(clf, n_features_to_select=6)
    new_data = sfs.fit_transform(df_data, labels)
    pd.DataFrame(new_data).to_csv("data_sfs.csv")
"""
Basic Machine Learning Models for accuracies
"""
# data_smote, labels_smote = over.fit_resample(new_data, labels)
#
# clf = KNeighborsClassifier(n_neighbors=3)
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\n3-NN-SMOTE  Accuracy = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\n3-NN-SMOTE  F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\n 3 Neighbours \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
#
# clf = KNeighborsClassifier(n_neighbors=10)
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\n10-NN-SMOTE Accuracy = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\n10-NN-SMOTE  F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\n 10 Neighbours \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
# clf = KNeighborsClassifier(n_neighbors=len(counts))
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\nAll_Classes-NN-SMOTE  Accuracy  = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\nAll_Classes-NN-SMOTE  F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\n All Class NN ( \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
# param_grid = [
#         {
#             'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#             'solver' : ['lbfgs', 'sgd', 'adam'],
#             'hidden_layer_sizes': [
#              (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
#              ]
#         }
#        ]
#
#
# clv = GridSearchCV(MLPClassifier(), param_grid, cv=cv_environment,
#                            scoring=scoring)
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\nMLP-SMOTE Accuracy  = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\nMLP-SMOTE F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\n MLP \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
# clf = RandomForestClassifier(n_estimators=10)
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\nRandomForest-SMOTE Accuracy  = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\nRandomForest-SMOTE F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\nRandom Forest \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
#
# clf = BaggingClassifier()
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\nBagging-SMOTE Accuracy  = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\nBagging-SMOTE F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\nBagging \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
#
# clf = SVC()
# results_smote = cross_validate(clf,  data_smote, labels_smote, cv = cv_environment,
#                               scoring=scoring)
#
# print('\nSVM-SMOTE Accuracy  = %.4f' % np.mean(results_smote['test_accuracy']))
# print('\nSVM-SMOTE F1 = %.4f' % np.mean(results_smote['test_f1_score']))
# file_save.write("\n\nSVM \n" + str(np.mean(results_smote['test_accuracy'])))
# file_save.write("\n" + str(np.mean(results_smote['test_f1_score'])))
#
# file_save.close()
#
