import os
import cv2
import mediapipe as mp
import csv
import numpy as np
from numpy import genfromtxt
import scipy as sp
from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt


def get_trajectory(csv_path, video_path):
    if os.path.isfile(csv_path):
        return

    camera = cv2.VideoCapture(video_path)
    mp_hands = mp.solutions.hands

    ret = True
    frame = 0
    fps = camera.get(cv2.CAP_PROP_FPS)
    while ret:
        ret, image = camera.read()
        if ret:
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # REVISAR QUE DEBE HACER SI NO ENCUENTRA MANO
                if not results.multi_hand_landmarks:
                    frame += 1
                    continue
                row = [float(frame) / fps]
                for finger in results.multi_hand_landmarks[0].landmark:
                    row.append(finger.x)
                    row.append(finger.y)

                with open(csv_path, mode='a') as f:
                    f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    f_writer.writerow(row)
            frame += 1


def get_calibration_distance(picture_path):
    image = cv2.imread(picture_path)
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_landmarks = results.multi_hand_landmarks[0]
        distance = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
    return distance


class Parkinson_movements:
    def __init__(self, csv_path, picture_path, movement, fps):
        self.mov = genfromtxt(csv_path, delimiter=',')
        self.movement = movement
        self.fs = fps
        self.organize_signal()
        self.calibrate(picture_path)
        # PADDING
        self.mov_pad = np.pad(self.mov, ((6, 6), (0, 0)), 'symmetric')

    def organize_signal(self):
        # Rearange matrix with trajectories
        x = range(1, 43, 2)
        y = range(2, 43, 2)
        permutation = [0, *x, *y]
        mov = self.mov[:, permutation]

        # Drop Z and separate time vector
        self.t0 = mov[:, 0]
        self.mov = np.delete(mov, 0, axis=1)

    def calibrate(self, picture_path):
        dist = get_calibration_distance(picture_path)
        self.mov = self.mov / dist

    def filter_signal(self):
        # HIGH PASS FILTER
        SOS = [[1, -2, 1, 1, -1.9446, 0.9530], [1, -2, 1, 1, -1.8686, 0.8767], [1, -2, 1, 1, -1.8273, 0.8353]]
        G = np.array([[0.9744], [0.9363], [0.9157], [1]])
        self.mov_filt = sp.signal.sosfiltfilt(SOS, self.mov_pad, axis=0, padtype=None) * np.prod(G)
        # plot_mov(t, mov_filt[7:-5], movement, ' filtro pasa-altas')

        # BAND PASS FILTER
        self.f_max, self.mov_fft, self.freq, self.idx_fmax = mov_freq(self.mov_filt, self.fs, self.movement)  # Main frequency of signal
        # print(self.f_max)
        f_min = max(0.0001, self.f_max - 1)
        Num = signal.firwin(11, [2 * (f_min) / self.fs, 2 * (self.f_max + 1) / self.fs],
                            pass_zero='bandpass')  # Design FIR filter
        # fir_freqz(Num, self.fs)
        mov_filter = sp.signal.filtfilt(Num, 1, self.mov_filt, axis=0, padtype=None, padlen=None, irlen=None)
        self.mov_filter = mov_filter[7:-5, :]  # Remove padding
        # plot_mov(t, mov_filter, movement, 'filtro pasa-bandas')

        # LOCAL MAXIMA
        idx_max = mov_localmax(self.t0, self.mov_filter, self.f_max, self.fs, self.movement, False)
        self.t1, self.mov_cut, self.idx_max = cut_mov(self.t0, self.mov_filter, idx_max, self.movement)

    def calc_speed(self):
        # SPEED CALCULATED AS DERIVATIVE
        self.speed = np.diff(self.mov_cut, axis=0)

    def calc_periods(self):
        # MOVEMENT PERIODICITY
        periods = periods_mov(self.t1, self.idx_max)
        plt.figure()
        plt.boxplot([periods[~np.isnan(periods[:, 25]), 25], periods[~np.isnan(periods[:, 29]), 29]])

    def calc_fft(self):
        # FFT
        self.mov_fft, self.freq = fft_mov(self.mov_cut, self.fs, True)

    def calc_amplitude(self):
        # AMPLITUDE TREND
        t_amp = self.t1
        amp_trend = detrend_mov(self.mov_cut, t_amp, self.movement, True)
        if self.movement == "fingertap":
            mov_amp1 = abs(self.mov_cut[:, 25])
            amp_trend1 = amp_trend[:, 25]
            mov_amp2 = abs(self.mov_cut[:, 29])
            amp_trend2 = amp_trend[:, 29]

        elif self.movement == "pronosup":
            mov_amp1 = abs(self.mov_cut[:, 4])
            amp_trend1 = amp_trend[:, 4]
            mov_amp2 = abs(self.mov_cut[:, 20])
            amp_trend2 = amp_trend[:, 20]

        else:
            mov_amp1 = abs(self.mov_cut[:, 25])
            amp_trend1 = amp_trend[:, 25]
            mov_amp2 = abs(self.mov_cut[:, 29])
            amp_trend2 = amp_trend[:, 29]

        return t_amp, mov_amp1, amp_trend1, mov_amp2, amp_trend2

    def get_plot(self):
        t1_graph = self.t0
        t2_graph = self.t1
        t3_graph = self.t1[1:]

        if self.movement == "fingertap":
            dedo1_o = self.mov[:, 25]
            dedo2_o = self.mov[:, 29]
            dedo1_seg = self.mov_cut[:, 25]
            dedo2_seg = self.mov_cut[:, 29]
            dedo1_vel = self.speed[:, 25]
            dedo2_vel = self.speed[:, 29]

        elif self.movement == "pronosup":
            dedo1_o = self.mov[:, 4]
            dedo2_o = self.mov[:, 20]
            dedo1_seg = self.mov_cut[:, 4]
            dedo2_seg = self.mov_cut[:, 20]
            dedo1_vel = self.speed[:, 4]
            dedo2_vel = self.speed[:, 20]

        else:
            dedo1_o = self.mov[:, 25]
            dedo2_o = self.mov[:, 29]
            dedo1_seg = self.mov_cut[:, 25]
            dedo2_seg = self.mov_cut[:, 29]
            dedo1_vel = self.speed[:, 25]
            dedo2_vel = self.speed[:, 29]

        return t1_graph, dedo1_o, dedo2_o, t2_graph, dedo1_seg, dedo2_seg, t3_graph, dedo1_vel, dedo2_vel

        # plot_mov(self.t0, self.mov, self.movement, 'Original')  # Plot original signal
        # plot_mov(self.t1, self.mov_cut, self.movement, 'segmentada') #Plot signal segmentada


def fft_mov(mov, fs, show):
    mov_rows = len(mov)
    fft2 = fft(mov, axis=0) / mov_rows
    fft1 = fft2[:(mov_rows // 2), :]
    fft1[1:-1, :] = 2 * fft1[1:- 1, :]
    xf = fftfreq(mov_rows, 1 / fs)[:mov_rows // 2]

    # Pulgar es el 25 e indice el 29
    # plt.plot(xf, np.abs(fft1[0:mov_rows//2, 29]))
    # plt.grid()
    # plt.show()

    if show:
        plt.figure()
        plt.imshow(np.transpose(abs(fft1[:, 0:21])), extent=[0, 15, 20, 0], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Articulación')
        plt.title('FFT - eje x')
        plt.yticks(np.arange(0, 21, dtype=np.int))
        # plt.show()

        plt.figure()
        plt.imshow(np.transpose(abs(fft1[:, 21:])), extent=[0, 15, 20, 0], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Articulación')
        plt.title('FFT - eje y')
        plt.yticks(np.arange(0, 21, dtype=np.int))
        # plt.show()

    return abs(fft1), xf


def mov_freq(mov, fs, movement):
    if movement == 'pronosup':
        finger = 4  # Pulgar en x
    elif movement == 'fingertap':
        finger = 29  # Pulgar en y (Indice en y no es detectado bien en todos los casos por mediapipe)
    else:
        finger = 29 # Indice en y

    [mov_fft, freq] = fft_mov(mov, fs, False)
    idx_fmax = np.argmax(mov_fft[:, finger])
    f_max = freq[idx_fmax]
    return f_max, mov_fft, freq, idx_fmax


def mov_localmax(t, mov, f_max, fps, movement, show):
    min_sep = int(0.66*max(fps / f_max, 4))
    # print(fps / f_max)
    # print(min_sep)
    idx_max = np.full(mov.shape, False)  # Logical array to store max locations
    for i in range(mov.shape[1]):
        # Find peaks idx for each column (finger trajectory in an axis)
        peaks, properties = signal.find_peaks(mov[:, i], distance=min_sep, height=0, prominence=0)
        idx_max[peaks[0:10], i] = True
        # prominences = properties["prominences"]
        # n_peaks = len(peaks)
        # if n_peaks != 0:
        #     if n_peaks > 10:
        #         # Take the 10 peaks with higher prominence
        #         sorted_peaks = [peak for _, peak in sorted(zip(prominences, peaks), reverse=True)]
        #         idx_max[sorted_peaks[0:10], i] = True
        #     else:
        #         idx_max[peaks, i] = True
    if show:
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
            plt.legend(['Pulgar', 'Max pulgar', 'Índice', 'Max indice'], loc='upper right')
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
            plt.legend(['Pulgar', 'Max pulgar', 'Meñique', 'Max meñique'], loc='upper right')
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
            plt.legend(['Pulgar', 'Max pulgar', 'Índice', 'Max indice'], loc='upper right')

    return idx_max


def cut_mov(t, mov, idx_max, movement):
    # Segment signal: first index is 7 samples before the first max and last index is 7 samples after last max
    if movement == 'pronosup':
        finger = 4  # Pulgar en x
    elif movement == 'fingertap':
        finger = 29  # Indice en y
    else:
        finger = 29 # Indice en y

    first_max = np.where(idx_max[:, finger])[0][0]
    no = int(max([first_max - 7, 0]))

    last_max = np.where(idx_max[:, finger])[-1][-1]
    nf = int(min([last_max + 7, mov.shape[0]]))

    t = t[no:nf]
    mov = mov[no:nf, :]
    idx_max = idx_max[no:nf, :]

    return t, mov, idx_max


def periods_mov(t, idx_max):
    columns = idx_max.shape[1]
    periods = np.empty([9, columns])
    periods[:] = np.NaN
    for i in range(columns):
        t_localmax = t[idx_max[:, i].astype(bool)]
        diff_t = np.diff(t_localmax)
        periods[0:len(diff_t), i] = diff_t
    return periods


def detrend_mov(mov, t, movement, show):
    mov_detrend = sp.signal.detrend(abs(mov), axis=0)
    amp_trend = abs(mov) - mov_detrend

    if show:
        if movement == 'fingertap':
            plt.figure()
            plt.plot(t, abs(mov[:, 25]), 'tab:blue')  # Thumb in y
            plt.plot(t, amp_trend[:, 25], color='tab:blue', linestyle=':')  # Trend in thumb
            plt.plot(t, abs(mov[:, 29]), color='tab:orange')  # Index in y
            plt.plot(t, amp_trend[:, 29], color='tab:orange', linestyle=':')  # Trend in index
            plt.grid()
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.title("Tendencias de las señales")
            plt.legend(['Pulgar', "Tendencia pulgar", 'Índice', "Tendencia indice"], loc='upper right')
            # plt.show()
        elif movement == 'pronosup':
            plt.figure()
            plt.plot(t, abs(mov[:, 4]), 'tab:blue')  # Thumb in x
            plt.plot(t, amp_trend[:, 4], color='tab:blue', linestyle=':')  # Trend in thumb
            plt.plot(t, abs(mov[:, 20]), color='tab:orange')  # Pinky in x
            plt.plot(t, amp_trend[:, 20], color='tab:orange', linestyle=':')  # Trend in index
            plt.grid()
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.title("Tendencias de las señales")
            plt.legend(['Pulgar', "Tendencia pulgar", 'Meñique', "Tendencia meñique"], loc='upper right')
            # plt.show()
        else:
            # Falta arreglar cuales columnas devuelve?
            plt.figure()
            plt.plot(t, abs(mov[:, 25]), 'tab:blue')  # Thumb in y
            plt.plot(t, amp_trend[:, 25], color='tab:blue', linestyle=':')  # Trend in thumb
            plt.plot(t, abs(mov[:, 29]), color='tab:orange')  # Pinky y
            plt.plot(t, amp_trend[:, 29], color='tab:orange', linestyle=':')  # Trend in index
            plt.grid()
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.title("Tendencias de las señales")
            plt.legend(['Pulgar', "Tendencia pulgar", 'Meñique', "Tendencia meñique"], loc='upper right')
            # plt.show()
    return amp_trend


def fir_freqz(num, fs):
    w, h = signal.freqz(num, 1, fs=fs)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
    #plt.show()