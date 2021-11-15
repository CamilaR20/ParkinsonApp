from feature_extraction import *
import os
from matplotlib import pyplot as plt


def plot_mov(t, mov, movement, title):
    if movement == 'fingertap':
        plt.figure()
        plt.plot(t, mov[:, 25])  # Thumb in y
        plt.plot(t, mov[:, 29])  # Index in y
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Finger tapping en y - ' + title)
        plt.legend(['Pulgar', 'Índice'], loc='upper right')
        #plt.show()
    elif movement == 'pronosup':
        plt.figure()
        plt.plot(t, mov[:, 4])  # Thumb in x
        plt.plot(t, mov[:, 20])  # Pinky in x
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Prono-supinación en x - ' + title)
        plt.legend(['Pulgar', 'Meñique'], loc='upper right')
        #return t, mov[:, 4], mov[:, 20]
        #plt.show()
    else:
        plt.figure()
        plt.plot(t, mov[:, 25])  # Thumb in y
        plt.plot(t, mov[:, 29])  # Index in y
        plt.plot(t, mov[:, 33])  # Middle in y
        plt.plot(t, mov[:, 37])  # Ring in y
        plt.plot(t, mov[:, 40])  # Pinky in y
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Fist open-close en y - ' + title)
        plt.legend(['Pulgar', 'Índice', 'Corazón', 'Anular', 'Meñique'], loc='upper right')
        #return t, mov[:, 25], mov[:, 41]
        #plt.show()


def plot_localmax(t, idx_max, mov, movement):
    if movement == 'fingertap':
        plt.figure()
        plt.plot(t, mov[:, 25], 'tab:blue')  # Thumb in y
        plt.plot(t[idx_max[:, 25]], mov[idx_max[:, 25], 25], color='tab:blue', ls='', marker='v')
        plt.plot(t, mov[:, 29], 'tab:orange')  # Index in y
        plt.plot(t[idx_max[:, 29]], mov[idx_max[:, 29], 29], color='tab:orange', ls='', marker='v')
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Finger tapping en y - Máximos Locales')
        plt.legend(['Pulgar', 'Max pulgar', 'Índice', 'Max indice'], loc='lower right')
        # plt.show()
    elif movement == 'pronosup':
        plt.figure()
        plt.plot(t, mov[:, 4], 'tab:blue')  # Thumb in y
        plt.plot(t[idx_max[:, 4]], mov[idx_max[:, 4], 4], color='tab:blue', ls='', marker='v')
        plt.plot(t, mov[:, 20], 'tab:orange')  # Index in y
        plt.plot(t[idx_max[:, 20]], mov[idx_max[:, 20], 20], color='tab:orange', ls='', marker='v')
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Pronosupinacion en x - Máximos Locales')
        plt.legend(['Pulgar', 'Max pulgar', 'Meñique', 'Max meñique'], loc='lower right')
        # plt.show()
    else:
        plt.figure()
        plt.plot(t, mov[:, 25], 'tab:blue')  # Thumb in y
        plt.plot(t[idx_max[:, 25]], mov[idx_max[:, 25], 25], color='tab:blue', ls='', marker='v')
        plt.plot(t, mov[:, 29], 'tab:orange')  # Index in y
        plt.plot(t[idx_max[:, 29]], mov[idx_max[:, 29], 29], color='tab:orange', ls='', marker='v')
        plt.grid()
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.title('Fist open-close en y - Máximos Locales')
        plt.legend(['Pulgar', 'Max pulgar', 'Índice', 'Max indice'], loc='lower right')


if __name__ == '__main__':
    main_path = '/Users/camilaroa/Downloads/ParkinsonVideos/0005/07-10-2021, 10-57, ON'
    movement = 'fist'
    finger = 'r'
    video_path = os.path.join(main_path, movement + '_' + finger + '.mp4')
    csv_path = os.path.join(main_path, movement + '_' + finger + '.csv')
    picture_path = os.path.join(main_path, movement + '_' + finger + '.jpg')
    fps = 30

    get_trajectory(csv_path, video_path)

    video_features = Parkinson_movements(csv_path, picture_path, movement, fps)
    video_features.filter_signal()
    plot_mov(video_features.t0, video_features.mov, movement, 'señal original')
    plot_mov(video_features.t0, video_features.mov_filt[7:-5], movement, ' filtro pasa-altas')
    plot_mov(video_features.t0, video_features.mov_filter, movement, 'filtro pasa-bandas')
    plot_mov(video_features.t1, video_features.mov_cut, movement, 'Señal segmentada')
    plot_localmax(video_features.t1, video_features.idx_max, video_features.mov_cut, movement)


    plt.figure()
    plt.plot(video_features.t1, video_features.mov_cut[:, -1])
    # plt.plot(t[idx_max], trajectory[idx_max], color='tab:blue', ls='', marker='v')
    plt.grid()
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    # plt.title(movement + '' + filename['Hand'] + ' - ' + str(idx))


    # video_features.calc_speed()
    # video_features.calc_periods()
    video_features.calc_fft()
    # video_features.calc_amplitude()

    plt.show()