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
from sklearn.metrics import roc_curve

class customMLP(MLPClassifier):
    def predict(self, X, threshold=0.49):
        result = super(customMLP, self).predict_proba(X)
        predictions = [1 if p > threshold else 0 for p in result[:, 1]]
        return predictions

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

    # idx = 5
    # X = np.delete(X, [idx, idx + 6, idx + 12, idx + 18, idx + 24, idx + 30], axis=1)
    X = np.delete(X, [-1, -2, -3], axis=1)
    # print(X.shape)

    # clf = LinearSVC(random_state=0, dual=False, tol=1e-5)
    # clf = SVC(random_state=0, tol=1e-5, kernel='rbf')
    # clf = SVC(random_state=0, tol=1e-5, kernel='poly', degree=3)
    # clf = KNeighborsClassifier(5)
    # clf = LinearDiscriminantAnalysis()
    # clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=0) # Neural net
    clf = customMLP(hidden_layer_sizes=(10,), max_iter=500, random_state=0)
    accuracy = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    sensitivity = cross_val_score(clf, X, y, cv=3, scoring='recall')
    balanced_acc = cross_val_score(clf, X, y, cv=3, scoring='balanced_accuracy')
    specificity = np.array([(2 *  b) - s for s, b in zip(sensitivity, balanced_acc)])

    print("%0.2f accuracy with a standard deviation of %0.2f" % (accuracy.mean(), accuracy.std()))
    print("%0.2f sensitivity with a standard deviation of %0.2f" % (sensitivity.mean(), sensitivity.std()))
    print("%0.2f specificity with a standard deviation of %0.2f" % (specificity.mean(), specificity.std()))
    # print("%0.2f balanced accuracy with a standard deviation of %0.2f" % (balanced_acc.mean(), balanced_acc.std()))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40, stratify=y)
    # clf.fit(X_train, y_train)

    # # Roc curve for MLP classifier
    # y_pred = clf.predict_proba(X_test)
    # # keep probabilities for the positive outcome only
    # y_pred = y_pred[:, 1]
    # # calculate roc curves
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # # calculate the g-mean for each threshold
    # gmeans = np.sqrt(tpr * (1 - fpr))
    # # locate the index of the largest g-mean
    # ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # # plot the roc curve for the model
    # plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # plt.plot(fpr, tpr, marker='.', label='Logistic')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # # axis labels
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend()
    # plt.title('ROC curve')
    # # show the plot
    # plt.show()

    # plt.figure()
    # plt.plot(clf.loss_curve_)
    # plt.xlabel('Número de iteraciones')
    # plt.ylabel('Pérdida')

    # print(clf.score(X_test, y_test))

    # predictions = clf.predict(X_test)
    # X_miss = X_test[y_test != predictions, :]
    # print(X_miss)

    # Confusion matrix
    # plt.figure()
    # disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=classes, cmap=plt.cm.Blues, normalize=None)
    # disp.ax_.set_title('Confusion Matrix')
    # plt.show()
