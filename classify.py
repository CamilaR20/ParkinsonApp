import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    fps = 30
    excel_file = '/Users/camilaroa/Downloads/videoList.xlsx'
    filenames_vid = pd.read_excel(excel_file, sheet_name='features', dtype={'PatientID': str})
    idx_drop = filenames_vid[(filenames_vid.Useful == 0)].index
    filenames_vid.drop(idx_drop, inplace=True)

    classes = ['Control', 'Parkinson']
    features = filenames_vid.drop(columns=['PatientID', 'DateTimeStatus', 'Movement', 'Hand', 'Useful']).to_numpy()

    X = features[:, 0:-1]
    # X = StandardScaler().fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)
    y = features[:, -1]

    clf = LinearSVC(random_state=0, dual=False, tol=1e-5)
    # clf = SVC(random_state=0, tol=1e-5, kernel='rbf')
    # clf = SVC(random_state=0, tol=1e-5, kernel='poly', degree=3)
    # clf = KNeighborsClassifier(8)
    # clf = GaussianNB()
    # clf = MLPClassifier(max_iter=700, random_state=0) # Neural net
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    X_train, X_test, y_train, y_test = train_test_split(features[:, :-1], features[:, -1], test_size=0.2, random_state=42, stratify=features[:, -1])
    clf.fit(X_train, y_train)

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=classes, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Confusion Matrix')

    # Features distribution plot
    # feature_names = ['Frequency', 'Periods STD', 'Amplitude x frequency', 'Amplitude trend slope', 'Energy fmax (%)']
    # plt.figure()
    # s = plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=y, cmap='prism')
    # plt.xlabel(feature_names[0])
    # plt.ylabel(feature_names[1])
    # plt.legend(*s.legend_elements(), loc="upper right", title="Classes")
    #
    # plt.figure()
    # s = plt.scatter(X[:, 2], X[:, 3], alpha=0.2, c=y, cmap='prism')
    # plt.xlabel(feature_names[2])
    # plt.ylabel(feature_names[3])
    # plt.legend(*s.legend_elements(), loc="upper right", title="Classes")
    #
    # plt.figure()
    # s = plt.scatter(X[:, 3], X[:, 4], alpha=0.2, c=y, cmap='prism')
    # plt.xlabel(feature_names[3])
    # plt.ylabel(feature_names[4])
    # plt.legend(*s.legend_elements(), loc="upper right", title="Classes")

    # Dataset classes distribution
    # plt.figure()
    # sns.countplot(y, palette=["#f7d754", "#87b212"])
    # plt.title("Distribución del dataset")

    # plt.show()

