import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from feature_extraction import *


def signal_periods(t, idx_max):
    t_localmax = t[idx_max]
    periods = np.diff(t_localmax)
    return periods


def signal_amplitudeTrend(t, sig):
    sig_detrend = sp.signal.detrend(abs(sig))
    amp_trend = abs(sig) - sig_detrend
    slope = (amp_trend[-1] - amp_trend[0]) / (t[-1] - t[0])
    return slope, amp_trend


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
        video_path = test_path + '.mp4'
        print(video_path)

        get_trajectory(csv_path, video_path)
        video_features = Parkinson_movements(csv_path, picture_path, movement, fps)
        video_features.filter_signal()
        video_features.calc_speed()

        t = video_features.t1
        trajectory = video_features.mov_cut[:, -1]
        trajectory_fft = video_features.mov_fft[:, -1]
        trajectory_freq = video_features.freq
        trajectory_speed = video_features.speed[:, -1]

        f_max = video_features.f_max
        idx_fmax = video_features.idx_fmax
        # print(f_max, trajectory_freq[idx_fmax])

        # Get local max
        idx_max = video_features.idx_max[:, -1]
        periods = signal_periods(t, idx_max)

        # Features
        t_std = 1 if len(periods) == 0 else np.std(periods) / np.mean(periods)
        axf = np.mean(trajectory[idx_max]) * f_max
        trend_slope, trend = signal_amplitudeTrend(t, trajectory)
        energy_fmax = trajectory_fft[idx_fmax] / np.sum(trajectory_fft)
        speed_slope, speed_trend = signal_amplitudeTrend(t[:-1], trajectory_speed)
        # Diferencia izquierda - derecha, relación entre amplitud y velocidad

        feature_vector.loc[idx, 'f_max'] = f_max
        feature_vector.loc[idx, 't_std'] = t_std
        feature_vector.loc[idx, 'axf'] = axf
        feature_vector.loc[idx, 'trend_slope'] = trend_slope
        feature_vector.loc[idx, 'energy_fmax'] = energy_fmax
        feature_vector.loc[idx, 'speed_slope'] = speed_slope
        feature_vector.loc[idx, 'bradykinesia'] = 1 - int(filename['PatientID'][0])

        # plt.figure()
        # plt.plot(t, trajectory)
        # plt.plot(t[idx_max], trajectory[idx_max], color='tab:blue', ls='', marker='v')
        # plt.plot(t, trend, color='tab:blue', linestyle=':')
        # plt.grid()
        # plt.xlabel('Tiempo (s)')
        # plt.ylabel('Amplitud')
        # plt.title(movement + '' + filename['Hand'] + ' - ' + str(idx))

        # plt.figure()
        # plt.plot(trajectory_freq, trajectory_fft)
        # plt.grid()
        # plt.xlabel('frecuencia (Hz)')
        # plt.ylabel('Amplitud')
        # plt.title(movement + '' + filename['Hand'] + ' - ' + str(idx))
        # plt.show()

    # print(feature_vector.head())
    feature_vector.to_excel(excel_file, sheet_name='features', index=False)

