import scipy as sp
from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def fft_mov(mov, show):
    mov_rows = len(mov)
    fs = 30
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
        plt.show()

        plt.figure()
        plt.imshow(np.transpose(abs(fft1[:, 21:])), extent=[0, 15, 20, 0], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Articulación')
        plt.title('FFT - eje y')
        plt.yticks(np.arange(0, 21, dtype=np.int))
        plt.show()

    return abs(fft1), xf


def mov_freq(mov):
    [mov_fft, freq] = fft_mov(mov, False)
    f_max = freq[np.argmax(mov_fft[:, 29])]
    return f_max


def mov_localmax(t, mov, f_max, movement, show):
    min_sep = 30 // f_max - 5
    idx_max = np.full(mov.shape, False)  # Logical array to store max locations
    for i in range(mov.shape[1]):
        # Find peaks idx for each column (finger trajectory in an axis)
        peaks, properties = signal.find_peaks(mov[:, i], distance=min_sep, height=0, prominence=0)
        prominences = properties["prominences"]
        n_peaks = len(peaks)
        if n_peaks != 0:
            if n_peaks > 10:
                # Take the 10 peaks with higher prominence
                sorted_peaks = [peak for _, peak in sorted(zip(prominences, peaks), reverse=True)]
                idx_max[sorted_peaks[0:10], i] = True
            else:
                idx_max[peaks, i] = True
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
    else:
        finger = 29  # Indice en y

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
        t_localmax = t[idx_max[:, i].astype(np.bool)]
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
            plt.legend(['Pulgar', "Tendencia pulgar", 'Índice',"Tendencia indice"], loc='upper right')
            #plt.show()
        elif movement == 'pronosup':
            plt.figure()
            plt.plot(t, abs(mov[:, 4]), 'tab:blue') #Thumb in x
            plt.plot(t, amp_trend[:, 4], color='tab:blue', linestyle=':')  # Trend in thumb
            plt.plot(t, abs(mov[:, 20]), color='tab:orange')  # Pinky in x
            plt.plot(t, amp_trend[:, 20], color='tab:orange', linestyle=':')  # Trend in index
            plt.grid()
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.title("Tendencias de las señales")
            plt.legend(['Pulgar', "Tendencia pulgar", 'Meñique', "Tendencia meñique"], loc='upper right')
            #plt.show()
        else:
            #Falta arreglar cuales columnas devuelve?
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
            #plt.show()
    return amp_trend


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
        plt.show()
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
