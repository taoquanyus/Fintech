import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("preprocesses/train_clean_encode.csv")
test = pd.read_csv("preprocesses/test_clean_encode.csv")

features = train.columns.tolist()
features.remove("CUST_UID")
features.remove("LABEL")
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)