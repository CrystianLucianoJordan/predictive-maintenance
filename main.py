

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import brier_score_loss, r2_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
# algorithm
from sklearn.cluster import DBSCAN
#mpld3.enable_notebook()
# Load Dataset
data = pd.read_csv("teste1.csv")

# clear dataset
data.dropna(axis=0, inplace=True)
dataBase = data.copy()
dataBase.drop(["misalignment", 'Timestamp'], axis=1, inplace=True)
dataBase = dataBase.values
flags = data["misalignment"]
flags = flags.values

# scaler
# Transformação do dado -> normalização das grandezas
scaler = MinMaxScaler()
normalisedBase = scaler.fit_transform(dataBase)

# outlier -> identificou
model = DBSCAN(eps = 5, min_samples = 10).fit(normalisedBase)
filteredBase = normalisedBase[model.labels_ != -1]
flags = flags[model.labels_ != -1]


# split train, test for calibration
dataBase_train, dataBase_test, flags_train, flags_test = train_test_split(filteredBase, flags, test_size=0.15, random_state=42)


# SVM
clf = svm.SVC() # instanciou um objeto de modelo do tipo SVR
clf.fit(dataBase_train, flags_train) # treinou o modelo
prob_pos_clf = clf.predict(dataBase_test) # testou o modelo

# scorer
clf_score = r2_score(flags_test, prob_pos_clf) # métrica de avaliação
print("r2: %1.3f" % clf_score)


# Best possible score is 1.0 and it can be negative 
# (because the model can be arbitrarily worse).
# A constant model that always predicts the expected value of y, disregarding the input features,
# would get a R^2 score of 0.0.


 #############################################################################
# Plot the data and the predicted probabilities
plt.figure()
y_unique = np.unique(flags)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))

pca = PCA(n_components=3).fit(dataBase_train)
x_pca = pca.transform(dataBase_train)

for this_y, color in zip(y_unique, colors):
    this_X = x_pca[flags_train == this_y]
    label = "Normal" if this_y == 0 else "Falha"
    plt.scatter(this_X[:, 0], this_X[:, 1],
                c=color[np.newaxis, :],
                alpha=0.5, edgecolor='k',
                label="Status %s" % label)
plt.legend(loc="best")
plt.title("Data")
