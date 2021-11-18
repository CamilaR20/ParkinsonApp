import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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


    X = MinMaxScaler().fit_transform(X)
    max_clusters = 5
    error = []
    feature_names = ['Fmax', 'Amplitude x Fmax', 'Energía Fmax (%)']
    # variances = np.std(X, axis=1)

    # plt.scatter(x, y, c=label_color)

    for i in range(2, max_clusters + 1):
        cluster = KMeans(n_clusters=i, random_state=0)
        cluster.fit(X)
        labels = cluster.labels_
        error.append(cluster.inertia_)

        fig1 = plt.figure(figsize=(20, 6))
        # fig1.suptitle('Golpeteo de dedos - Etiquetas reales')
        fig1.suptitle('Golpeteo de dedos - n_clusters = ' + str(i))
        ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 2], X[:, 4], alpha=1, c=labels, cmap='Set1')
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_zlabel(feature_names[2])
        ax1.set_title('Mano derecha')
        ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(X[:, 6], X[:, 8], X[:, 10], alpha=1, c=labels, cmap='Set1')
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_zlabel(feature_names[2])
        ax2.set_title('Mano izquierda')

        # fig1 = plt.figure(figsize=(20, 8))
        # ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
        # ax1.scatter(X[:, 12], X[:, 14], X[:, 16], alpha=1, c=labels, cmap='Set1')
        # ax1.set_xlabel(feature_names[0])
        # ax1.set_ylabel(feature_names[1])
        # ax1.set_zlabel(feature_names[2])
        # ax1.set_title('Prono-supinación, derecha, ' + str(i))
        # ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
        # ax2.scatter(X[:, 18], X[:, 20], X[:, 22], alpha=1, c=labels, cmap='Set1')
        # ax2.set_xlabel(feature_names[0])
        # ax2.set_ylabel(feature_names[1])
        # ax2.set_zlabel(feature_names[2])
        # ax2.set_title('Prono-supinación, izquierda')

        # fig1 = plt.figure(figsize=(20, 8))
        # ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
        # ax1.scatter(X[:, 24], X[:, 26], X[:, 28], alpha=1, c=labels, cmap='Set1')
        # ax1.set_xlabel(feature_names[0])
        # ax1.set_ylabel(feature_names[1])
        # ax1.set_zlabel(feature_names[2])
        # ax1.set_title('Cierre de puño, derecha, ' + str(i))
        # ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
        # ax2.scatter(X[:, 30], X[:, 32], X[:, 34], alpha=1, c=labels, cmap='Set1')
        # ax2.set_xlabel(feature_names[0])
        # ax2.set_ylabel(feature_names[1])
        # ax2.set_zlabel(feature_names[2])
        # ax2.set_title('Cierre de puño, izquierda')
        #
        # fig1 = plt.figure(figsize=(6, 6))
        # ax1 = fig1.add_subplot(projection='3d')
        # ax1.scatter(X[:, -3], X[:, -2], X[:, -1], alpha=1, c=labels, cmap='Set1')
        # ax1.set_xlabel('Golpeteo de dedos')
        # ax1.set_ylabel('Prono-supinación')
        # ax1.set_zlabel('Cierre de puño')
        # ax1.set_title('Diferencia entre derecha e izquierda')

        plt.show()


    print(error)
