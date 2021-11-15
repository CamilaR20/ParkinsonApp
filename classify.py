import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    fps = 30
    excel_file = '/Users/camilaroa/Downloads/videoList.xlsx'
    filenames_vid = pd.read_excel(excel_file, sheet_name='features', dtype={'PatientID': str})
    idx_drop = filenames_vid[(filenames_vid.Useful == 0)].index
    filenames_vid.drop(idx_drop, inplace=True)

    features = filenames_vid.drop(columns=['PatientID', 'DateTimeStatus', 'Movement', 'Hand']).to_numpy()


    X = features[:, 0:-1]
    # X = StandardScaler().fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)
    y = features[:, -1]

    clf = LinearSVC(random_state=0, dual=False, tol=1e-5)
    # clf = SVC(random_state=0, tol=1e-5, kernel='rbf')
    # clf = KNeighborsClassifier(1)
    # clf = GaussianNB()
    # clf = MLPClassifier(max_iter=1000) # Neural net
    scores = cross_val_score(clf, X, y, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # X_train, X_test, y_train, y_test = train_test_split(features[:, :-1], features[:, -1], test_size=0.1, random_state=42, stratify=features[:, -1])

    # plt.figure()
    # sns.countplot(y, palette=["#f7d754", "#87b212"])
    # plt.title("Distribución del dataset")
    # plt.show()


