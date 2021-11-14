import numpy as np
import pandas as pd
import os
from feature_extraction import *


def signal_localmax(t, sig, f_max, fps):
    min_sep = int(0.66*max(fps / f_max, 4))
    idx_max = np.full(sig.shape, False)  # Logical array to store max locations
    # Find peaks idx for each column (finger trajectory in an axis)
    peaks, properties = signal.find_peaks(sig, distance=min_sep, height=0, prominence=0)
    idx_max[peaks[0:10]] = True
    return idx_max


def signal_periods(t, idx_max):
    t_localmax = t[idx_max.astype(bool)]
    periods = np.diff(t_localmax)
    return periods


def signal_amplitudeTrend(t, sig):
    sig_detrend = sp.signal.detrend(abs(sig))
    amp_trend = abs(sig) - sig_detrend
    slope = (amp_trend[-1] - amp_trend[0]) / (t[-1] - t[0])
    return slope


if __name__ == '__main__':
    fps = 30
    excel_file = '/Users/camilaroa/Downloads/videoList.xlsx'
    filenames_vid = pd.read_excel(excel_file, sheet_name='features', dtype={'PatientID': str})
    feature_vector = filenames_vid.copy()

    folder_vid = '/Users/camilaroa/Downloads/ParkinsonVideos'

    for idx, filename in filenames_vid.iterrows():
        movement = filename['Movement']
        test_path = os.path.join(folder_vid, filename['PatientID'], filename['DateTimeStatus'], filename['Movement'] + '_' + filename['Hand'])
        csv_path = test_path + '.csv'
        picture_path = test_path + '.jpg'

        try:
            video_features = Parkinson_movements(csv_path, movement, fps)
        except:
            break
        video_features.filter_signal()

        if movement == 'fingertap':
            trajectory = video_features.mov_cut[:, 29] - video_features.mov_cut[:, 25]
            trajectory_fft = video_features.mov_fft[:, 29] - video_features.mov_fft[:, 25]
        elif movement == 'pronosup':
            trajectory = video_features.mov_cut[:, 4] - video_features.mov_cut[:, 20]
            trajectory_fft = video_features.mov_fft[:, 4] - video_features.mov_fft[:, 20]
        else:
            trajectory = (video_features.mov_cut[:, 25] + video_features.mov_cut[:, 29] + video_features.mov_cut[:, 33] + video_features.mov_cut[:, 37] + video_features.mov_cut[:, 40]) / 5
            trajectory_fft = (video_features.mov_fft[:, 25] + video_features.mov_fft[:, 29] + video_features.mov_fft[:, 33] + video_features.mov_fft[:, 37] + video_features.mov_fft[:, 40]) / 5

        t = video_features.t1
        # Remove DC
        trajectory = trajectory - np.mean(trajectory)

        f_max = video_features.f_max
        trajectory_freq = video_features.freq
        idx_fmax = video_features.idx_fmax

        # Get local max
        idx_max = signal_localmax(t, trajectory, f_max, fps)

        # Features
        t_std = np.std(signal_periods(t, idx_max))
        axf = np.mean(trajectory[idx_max]) * f_max
        trend_slope = signal_amplitudeTrend(t, trajectory)
        energy_fmax = trajectory_fft[idx_fmax] / np.sum(trajectory_fft)

        feature_vector.loc[idx, 'f_max'] = f_max
        feature_vector.loc[idx, 't_std'] = t_std
        feature_vector.loc[idx, 'axf'] = axf
        feature_vector.loc[idx, 'trend_slope'] = trend_slope
        feature_vector.loc[idx, 'energy_fmax'] = energy_fmax

        plt.figure()
        plt.plot(t, trajectory)
        plt.plot(t[idx_max], trajectory[idx_max], color='tab:blue', ls='', marker='v')
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title(movement + '' + filename['Hand'] + ' - ' + str(idx))

    # print(feature_vector.head())
    feature_vector.to_excel(excel_file, sheet_name='features', index=False)

    plt.show()

