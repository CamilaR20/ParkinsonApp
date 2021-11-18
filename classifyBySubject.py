import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    # filenames_vid.drop(idx_drop, inplace=True)

    classes = ['Control', 'Parkinson']
    features = filenames_vid.drop(columns=['PatientID', 'DateTimeStatus', 'Movement', 'Hand', 'Useful']).to_numpy()
    feature_names = ['f_max', 't_std', 'axf', 'trend_slope', 'energy_fmax', 'speed_slope']

    X = np.zeros([features.shape[0] // 6, ((features.shape[1] - 1) * 6) + 3])
    y = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        y[i] = features[i * 6, -1]
        dif_lr_fingertap = features[i * 6, 2] / features[(i * 6) + 1, 2]
        dif_lr_pronosup = features[(i * 6) + 2, 2] / features[(i * 6) + 3, 2]
        dif_lr_fist = features[(i * 6) + 4, 2] / features[(i * 6) + 5, 2]

        X[i, :] = np.concatenate((features[i * 6, :-1], features[(i * 6) + 1, :-1], features[(i * 6) + 2, :-1],
                                  features[(i * 6) + 3, :-1], features[(i * 6) + 4, :-1], features[(i * 6) + 5, :-1],
                                  [dif_lr_fingertap, dif_lr_pronosup, dif_lr_fist]), axis=0)

    # y = np.transpose(y)

    clf = LinearSVC(random_state=0, dual=False, tol=1e-5)
    # clf = SVC(random_state=0, tol=1e-5, kernel='rbf')
    # clf = SVC(random_state=0, tol=1e-5, kernel='poly', degree=3)
    # clf = KNeighborsClassifier(8)
    # clf = LinearDiscriminantAnalysis()
    # clf = MLPClassifier(max_iter=700, random_state=0) # Neural net
    scores = cross_val_score(clf, X, y, cv=3)
    print(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=classes, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Confusion Matrix')
    plt.show()
