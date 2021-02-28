import pandas as pd
import numpy as np
import os
os.chdir("C:/Users/Joe/PycharmProjects/Data_science_Volcano_code")
# Import the classifier models.
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv("data/train/513181.csv")
classes = list(range(0,1000000000,1000))
print(len(classes))
classifiers = ('1-nn', '3-nn', '5-nn','Decision Tree', 'MLP', 'Bagging', 'Random subspace', 'Random Forest')
acc = np.empty(len(classifiers))