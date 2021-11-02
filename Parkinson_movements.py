import scipy as sp
from numpy import genfromtxt
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from extraccion import *

class Parkinson_movements:
    def __init__(self, filename, movement):
        #mov_num = '07l'
        #filename = ('./mov_csv/fingertap_' + mov_num + '.csv')
        self.movement = movement
        self.mov = genfromtxt(filename, delimiter=',')

    def organize_signal(self):
        self.fs = 30
        # Rearange matrix with trajectories
        x = range(1, 64, 3)
        y = range(2, 64, 3)
        z = range(3, 64, 3)
        permutation = [0, *x, *y, *z]
        mov = self.mov[:, permutation]
        #mov[:, 0] = mov[:, 0] / 1000

        # Drop Z and separate time vector
        self.t0 = mov[:, 0]
        self.mov = np.delete(mov, slice(41, 63), axis=1)
        self.mov = np.delete(self.mov, 0, axis=1)

        # PADDING
        self.mov_pad = np.pad(self.mov, ((6, 6), (0, 0)), 'symmetric')

    def filtered_signal(self):
        # HIGH PASS FILTER
        SOS = [[1, -2, 1, 1, -1.9446, 0.9530], [1, -2, 1, 1, -1.8686, 0.8767], [1, -2, 1, 1, -1.8273, 0.8353]]
        G = np.array([[0.9744], [0.9363], [0.9157], [1]])
        self.mov_filt = sp.signal.sosfiltfilt(SOS, self.mov_pad, axis=0, padtype=None) * np.prod(G)
        # plot_mov(t, mov_filt[7:-5], movement, ' filtro pasa-altas')

        # BAND PASS FILTER
        f_max = mov_freq(self.mov_filt)  # Main frequency of signal
        f_min = max(0.0001, f_max-1)
        Num = signal.firwin(9, [2 * (f_min) / self.fs, 2 * (f_max + 1) / self.fs], pass_zero='bandpass')  # Design FIR filter
        # fir_freqz(Num, Fs)
        mov_filter = sp.signal.filtfilt(Num, 1, self.mov_filt, axis=0, padtype=None, padlen=None, irlen=None)
        self.mov_filter = mov_filter[7:-5, :]  # Remove padding
        # plot_mov(t, mov_filter, movement, 'filtro pasa-bandas')

        # LOCAL MAXIMA
        idx_max = mov_localmax(self.t0, self.mov_filter, f_max, self.movement, False)
        self.t1, self.mov_cut, self.idx_max = cut_mov(self.t0, self.mov_filter, idx_max, self.movement)

    def calc_speed(self):
        # SPEED CALCULATED AS DERIVATIVE
        self.speed = np.diff(self.mov_cut, axis=0)

    def periodicidad(self):
        # MOVEMENT PERIODICITY
        periods = periods_mov(self.t1, self.idx_max)
        plt.figure()
        plt.boxplot([periods[~np.isnan(periods[:, 25]), 25], periods[~np.isnan(periods[:, 29]), 29]])
        plt.title("Mediana y distribución de la periodicidad - fingertap")
        plt.xticks([1, 2], ['Pulgar', 'Índice'])
        plt.ylabel('Distribución')

    def calc_fft(self):
        #FFT
        self.mov_fft, self.freq = fft_mov(self.mov_cut, True)

    def calc_amplitud(self):
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

    def plotear_graficas(self):
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
        # plot_mov(self.t1[1:], self.speed, self.movement, 'velocidad') #Plot speed as derivative