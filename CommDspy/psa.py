import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, welch


def psa(signal, fs, rbw=None, mode='dB', obw=None, r=50, osr=4, window='blackmanharris', overlap=0.75, fullscale=1, sb_mode='DSB', plot=False):
    """
    Power Spectrum Analyzer. Estimates the power spectral density of a discrete-time signal vector using a welch's
    method, modified periodogram.
    :param signal: input signal
    :param fs: sampling frequency
    :param rbw: Resolution BandWidth
    :param mode: signal analyzer normalization mode:
        'dBm'             -  signal power spectrum (default)
        'dBm/Hz'          -  power spectral density (noise level measurement)
        'dBc'             -  signal level relative to the peak (mask measurment)
        'dBuv'            -  signal voltage spectrum
        'dBuv/sqrt(Hz)'   -  voltage spectral density (noise level measurement)
        'dBc/Hz '         -  signal level density (phase noise measurement)
        'dB'          -  skip normalization (same as Matlab PSD defaults)
        'dBFs'            -  for backoff calculations with respect to full scale
    :param obw: occupied bandwidth. Default value is None, if this is not None, the mode is considered to be 'dBc'
                regardless of the input
    :param r: load resistance in [Ohm]
    :param osr: Over Sampling Rate, standing for "Nfft/RBW" ratio
    :param window: window used for the PSA, default is blackman-harris
    :param overlap: amount of overlap for the blackman-harris window
    :param fullscale: used by the 'dBFs' mode, indicating the full scale
    :param sb_mode: Side Band mode, can be either:
        - 'DSB': double side band, reflecting the negative frequencies energy integration, on output will give the
                 energy of the entire signal
        - 'SSB': Single Side Band, omits negative frequencies in the integration, outputs half of the energy of the
                 signal
    :param plot: Boolean. If True, plotting the estimated PSD
    :return:
        f - frequency vector
        p_xx - power density of the given frequencies

    NOTE - For all power measurements (dBm) signal samples are assumed to be given in Volts over 50 Ohm load unless
           otherwise specified.
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    factor = 6.3  # 8 [dB]
    signal_flat = np.reshape(signal, -1)
    if obw is not None:
        mode = 'dBc'
    if rbw is None:
        rbw = np.max([fs/1000, 2*fs/len(signal_flat)/10])
    effective_fs = fs / 2 if np.all(np.isreal(signal)) and sb_mode == 'DSB' else fs
    # --------------------------------------------------------------------------------------------------------------
    # defining the normalization factor
    # --------------------------------------------------------------------------------------------------------------
    if mode == 'dBm':
        norm_factor = np.sqrt(1000 / r) * np.sqrt(rbw / effective_fs)
    elif mode == 'dBm/Hz':
        norm_factor = np.sqrt(1000/r/effective_fs)
    elif mode == 'dBuv':
        norm_factor = np.sqrt(rbw/effective_fs) * 1e6
    elif mode == 'dBuv/sqrt(Hz)':
        norm_factor = np.sqrt(1 / effective_fs) * 1e6
    elif mode == 'dBc':
        vrms = np.sqrt(np.mean(signal_flat * np.conj(signal_flat)))
        if obw is None:
            obw = rbw
        norm_factor = np.sqrt(obw / effective_fs) / vrms
    elif mode == 'dBFs':
        if obw is None:
            obw = rbw
        norm_factor = np.sqrt(obw / effective_fs) / fullscale
    elif mode == 'dB':
        norm_factor = 1
    else:
        raise ValueError('mode is not supported')
    # ==================================================================================================================
    # Computing window parameters
    # ==================================================================================================================
    window_n      = int(1.9 * np.ceil(fs / rbw))
    window_actual = get_window(window, window_n)
    # ==================================================================================================================
    # Computing Power Spectral Density parameters
    # ==================================================================================================================
    nfft      = int(2 ** (np.ceil(np.log2(window_n * osr))))
    n_overlap = int(np.floor(window_n * overlap))
    k         = np.fix((len(signal_flat) - n_overlap) / (window_n - n_overlap))
    # norm_factor *= k * np.linalg.norm(window_actual)**2 * factor
    # ==================================================================================================================
    # Computing power spectrum estimation
    # ==================================================================================================================
    fx, pxx = welch(signal_flat * norm_factor, nfft=nfft, fs=fs, window=window_actual, noverlap=n_overlap, scaling='spectrum')
    if plot:
        plt.figure()
        plt.plot(fx, 10 * np.log10(pxx))
        plt.title('Power Spectrum Analysis result')
        plt.ylabel(mode)
        plt.xlabel('Freq [Hz]')
        plt.grid()
        plt.show()

    return fx, pxx