import os
from glob import glob

import numpy as np
import pandas as pd

os.chdir("C:/Users/Joe/PycharmProjects/Data_science_Volcano_code")

# Import the classifier models.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PATH = "data\\train"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
# Define a list of classes (time in seconds to erupt) in bins of 100 from 0 to 1,000,000,000
classes = list(range(0, 1000000000, 100))
list_of_data = []
for file in all_csv_files:
    print(file)
    list_of_data = (file for _ in pd.read_csv(file, chunksize=10000, error_bad_lines=False, dtype=float,
                                              sep=",", header=0, warn_bad_lines=True))

labels = pd.read_csv("data\\train.csv")
labels['time_to_eruption'] = labels['time_to_eruption'].round(decimals=-2)

classifiers = ('1-nn', '3-nn', '5-nn', 'Decision Tree', 'MLP', 'Bagging', 'Random subspace', 'Random Forest')
acc = np.empty(len(classifiers))


def testing(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=101)

    # Training of random forest
    trainedforest = RandomForestClassifier(n_estimators=700).fit(x_train, y_train)
    predictionforest = trainedforest.predict(x_test)
    print(predictionforest)
