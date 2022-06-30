import CommDspy as cdsp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import json

def tx_example():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    prbs_type       = cdsp.constants.PrbsEnum.PRBS13
    bits_per_symbol = 2
    bit_order_inv   = False
    inv_msb         = False
    inv_lsb         = False
    pn_inv          = False
    constellation   = cdsp.constants.ConstellationEnum.PAM4
    full_scale      = True
    gray_coding     = False
    poly_coeff      = cdsp.get_polynomial(prbs_type)
    init_seed       = np.array([1] * prbs_type.value)
    # prbs_len        = 8191  # can be any number
    prbs_len        = 8191 + 1000  # 8191  # can be any number
    # ==================================================================================================================
    # Creating reference pattern
    # ==================================================================================================================
    # --------------------------------------------------------------------------------------------------------------
    # Getting PRBS binary pattern
    # --------------------------------------------------------------------------------------------------------------
    prbs_seq, seed_dut = cdsp.tx.prbs_gen(poly_coeff, init_seed, prbs_len * bits_per_symbol)
    # --------------------------------------------------------------------------------------------------------------
    # Duplicating if needed and coding
    # --------------------------------------------------------------------------------------------------------------
    pattern       = cdsp.tx.bin2symbol(prbs_seq, 2 ** bits_per_symbol, bit_order_inv, inv_msb, inv_lsb, pn_inv)
    pattern       = cdsp.tx.mapping(pattern, constellation, full_scale) if not gray_coding else cdsp.tx.mapping(cdsp.tx.coding_gray(pattern, constellation), constellation, full_scale)

    return pattern

def channel_example(pulse_show=False, pulse_eye=False,  awgn_eye=False, awgn_ch_eye=False):
    rolloff = 0.9
    pattern = tx_example()
    # ==================================================================================================================
    # Pulse shaping
    # ==================================================================================================================
    if pulse_show:
        tx_out_rect = cdsp.channel.pulse_shape(pattern,osr=32, span=8, method='rect')
        tx_out_sinc = cdsp.channel.pulse_shape(pattern,osr=32, span=8, method='sinc')
        tx_out_rcos = cdsp.channel.pulse_shape(pattern,osr=32, span=8, method='rcos', beta=rolloff)
        tx_out_rrc  = cdsp.channel.pulse_shape(pattern,osr=32, span=8, method='rrc', beta=rolloff)
        plt.figure()
        plt.plot(np.arange(0, len(tx_out_rect))/32, tx_out_rect, '-')
        plt.plot(np.arange(0, len(tx_out_rect))/32, tx_out_sinc, '-')
        plt.plot(np.arange(0, len(tx_out_rect))/32, tx_out_rcos, '-')
        plt.plot(np.arange(0, len(tx_out_rect))/32, tx_out_rrc, '-')
        plt.plot(pattern[8:-8], 'o')
        plt.grid()
        plt.legend(['Rect pulse',
                    'Sinc pulse',
                    f'Raised cosine pulse, (beta = {rolloff})',
                    f'RRC pulse, (beta = {rolloff})',
                    'Original symbols'])
        plt.title('Example of simple Tx prbs + pulse shaping')
        plt.xlabel(' Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # Eye diagram
    # ==================================================================================================================
    if pulse_eye:
        tx_out_rcos = cdsp.channel.pulse_shape(pattern,osr=32, span=8, method='rcos', beta=rolloff)
        plt.figure()
        eye_d, amp_vec = cdsp.eye_diagram(tx_out_rcos, 32, 128, fs_value=3, quantization=1024, logscale=True)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=1000, cmap='gray')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # AWGN channel
    # ==================================================================================================================
    if awgn_eye:
        rolloff = 0.9
        snr     = 10
        pattern = tx_example()
        ch_out = cdsp.channel.awgn(pattern, osr=32, span=8, method='rcos', beta=rolloff, snr=snr)
        eye_d, amp_vec = cdsp.eye_diagram(ch_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100,cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram, AWGN noise + pulse with SNR of {snr} [dB]')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # AWGN + ISI channel
    # ==================================================================================================================
    if awgn_ch_eye:
        rolloff = 0.9
        snr = 10
        b = [0.5]
        a = [1, -0.2]
        pattern = tx_example()
        ch_out = cdsp.channel.awgn_channel(pattern, b, a, osr=32, span=8, method='rcos', beta=rolloff, snr=snr)
        eye_d, amp_vec = cdsp.eye_diagram(ch_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram, ISI + AWGN noise + pulse with SNR of {snr} [dB]')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()

def rx_example(ch_out_eye=False, show_ctle=False, ctle_out_eye=False, rx_ffe_eye=False):
    # ==================================================================================================================
    # Tx + Channel setting
    # ==================================================================================================================
    prbs_type     = cdsp.constants.PrbsEnum.PRBS13
    constellation = cdsp.constants.ConstellationEnum.PAM4
    full_scale    = True
    rolloff = 0.9
    snr     = 10
    osr     = 32
    pattern = tx_example()
    # ==================================================================================================================
    # CTLE settings
    # ==================================================================================================================
    zeros   = [5e8, 11e9]
    poles   = [1e9, 20e9, 25e9]
    dc_gain = -4  # [dB]
    fs      = 53.125e9
    # ==================================================================================================================
    # Rx FFE settings
    # ==================================================================================================================
    ffe_precursors  = 4
    ffe_postcursors = 23
    ffe_len         = ffe_postcursors + ffe_precursors + 1
    dfe_taps        = 0
    # ==================================================================================================================
    # Loading data
    # ==================================================================================================================
    f = open(os.path.join('..', 'test_data', 'example_channel_full.json'))
    data = json.load(f)
    f.close()
    channel_sampled = data['channel_sampled']
    # ==================================================================================================================
    # Passing through channel
    # ==================================================================================================================
    ch_out = cdsp.channel.awgn_channel(pattern, channel_sampled, [1], osr=osr, span=8, method='rcos', beta=rolloff, snr=snr)
    if ch_out_eye:
        eye_d, amp_vec = cdsp.eye_diagram(ch_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram, loaded channel + pulse with SNR of {snr} [dB]')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # Passing through CTLE
    # ==================================================================================================================
    if show_ctle:
        numerator_dis, denomenator_dis = cdsp.rx.get_ctle_filter(zeros, poles, dc_gain, fs=fs, osr=osr)
        w_dis, h_dis = signal.freqz(numerator_dis, denomenator_dis, worN=np.logspace(7, 11, 1000), fs=fs*osr * 2 * np.pi)
        plt.semilogx(w_dis[w_dis < fs], 20 * np.log10(abs(h_dis[w_dis < fs])))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude response [dB]')
        plt.title('Example CTLE transfer function')
        plt.grid()
        plt.show()
    ctle_out = cdsp.rx.ctle(ch_out, zeros, poles, dc_gain, fs=fs, osr=osr)
    if ctle_out_eye:
        eye_d, amp_vec = cdsp.eye_diagram(ctle_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram,  loaded channel + pulse with SNR of {snr} [dB] ; after CTLE ')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # Estimating optimal Rx FFE and passing data through
    # ==================================================================================================================
    ctle_out_mat = cdsp.buffer(ctle_out, osr, 0)
    rx_ffe       = np.zeros(ffe_len)
    err          = float('inf')
    phase        = -1
    for ii, sampled_phase_data in enumerate(ctle_out_mat.T):
        rx_ffe_cand = cdsp.equalization_estimation_prbs(prbs_type, sampled_phase_data, constellation,
                                                        prbs_full_scale=full_scale,
                                                        ffe_postcursor=ffe_postcursors,
                                                        ffe_precursor=ffe_precursors,
                                                        dfe_taps=dfe_taps,
                                                        normalize=False,
                                                        bit_order_inv=False,
                                                        pn_inv_precoding=False,
                                                        gray_coded=False,
                                                        pn_inv_postmapping=False)
        if rx_ffe_cand[-1] < err:
            err    = rx_ffe_cand[-1]
            rx_ffe = rx_ffe_cand[0]
            phase  = ii
    # --------------------------------------------------------------------------------------------------------------
    # Passing through the Rx FFE
    # --------------------------------------------------------------------------------------------------------------
    rx_ffe_ups = cdsp.upsample(rx_ffe, osr)
    rx_ffe_out = signal.lfilter(rx_ffe_ups, 1, ctle_out)[ffe_len*osr:]
    if rx_ffe_eye:
        eye_d, amp_vec = cdsp.eye_diagram(rx_ffe_out, osr, 4*osr, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram,  loaded channel + pulse with SNR of {snr} [dB] ; after Rx FFE ')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()

    rx_ffe_out_mat = cdsp.buffer(rx_ffe_out, osr, 0)
    rx_ffe_out_ui = signal.lfilter(rx_ffe, 1, ctle_out_mat.T[phase, :])
    # return  rx_ffe_out_ui[len(rx_ffe):]
    return  rx_ffe_out_mat[:, phase]

def rx_example2(ch_out_eye=False, show_ctle=False, ctle_out_eye=False, rx_slicer_in_eye=False):
    # ==================================================================================================================
    # Tx + Channel setting
    # ==================================================================================================================
    prbs_type     = cdsp.constants.PrbsEnum.PRBS13
    constellation = cdsp.constants.ConstellationEnum.PAM4
    full_scale    = True
    rolloff = 0.9
    snr     = 10
    osr     = 32
    pattern = tx_example()
    # ==================================================================================================================
    # CTLE settings
    # ==================================================================================================================
    zeros   = [5e8, 11e9]
    poles   = [1e9, 20e9, 25e9]
    dc_gain = -4  # [dB]
    fs      = 53.125e9
    # ==================================================================================================================
    # Rx FFE settings
    # ==================================================================================================================
    ffe_precursors  = 4
    ffe_postcursors = 4
    ffe_len         = ffe_postcursors + ffe_precursors + 1
    dfe_taps        = 4
    # ==================================================================================================================
    # Loading data
    # ==================================================================================================================
    f = open(os.path.join('..', 'test_data', 'example_channel_full.json'))
    data = json.load(f)
    f.close()
    channel_sampled = data['channel_sampled']
    # ==================================================================================================================
    # Passing through channel
    # ==================================================================================================================
    ch_out = cdsp.channel.awgn_channel(pattern, channel_sampled, [1], osr=osr, span=8, method='rcos', beta=rolloff, snr=snr)
    ch_out = ch_out[len(channel_sampled):]
    if ch_out_eye:
        eye_d, amp_vec = cdsp.eye_diagram(ch_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram, loaded channel + pulse with SNR of {snr} [dB]')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # Passing through CTLE
    # ==================================================================================================================
    if show_ctle:
        numerator_dis, denomenator_dis = cdsp.rx.get_ctle_filter(zeros, poles, dc_gain, fs=fs, osr=osr)
        w_dis, h_dis = signal.freqz(numerator_dis, denomenator_dis, worN=np.logspace(7, 11, 1000), fs=fs*osr * 2 * np.pi)
        plt.semilogx(w_dis[w_dis < fs], 20 * np.log10(abs(h_dis[w_dis < fs])))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude response [dB]')
        plt.title('Example CTLE transfer function')
        plt.grid()
        plt.show()
    ctle_out = cdsp.rx.ctle(ch_out, zeros, poles, dc_gain, fs=fs, osr=osr)
    if ctle_out_eye:
        eye_d, amp_vec = cdsp.eye_diagram(ctle_out, 32, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram,  loaded channel + pulse with SNR of {snr} [dB] ; after CTLE ')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()
    # ==================================================================================================================
    # Estimating optimal Rx FFE and passing data through
    # ==================================================================================================================
    ctle_out_mat = cdsp.buffer(ctle_out, osr, 0)
    rx_ffe       = np.zeros(ffe_len)
    rx_dfe       = np.zeros(dfe_taps)
    err          = float('inf')
    phase        = -1
    for ii, sampled_phase_data in enumerate(ctle_out_mat.T):
        # if ii < 6:
        #     continue
        rx_ffe_dfe_cand = cdsp.equalization_estimation_prbs(prbs_type, sampled_phase_data, constellation,
                                                            prbs_full_scale=full_scale,
                                                            ffe_postcursor=ffe_postcursors,
                                                            ffe_precursor=ffe_precursors,
                                                            dfe_taps=dfe_taps,
                                                            normalize=False,
                                                            bit_order_inv=False,
                                                            pn_inv_precoding=False,
                                                            gray_coded=False,
                                                            pn_inv_postmapping=False)
        if rx_ffe_dfe_cand[-1] < err:
            err    = rx_ffe_dfe_cand[-1]
            rx_ffe = rx_ffe_dfe_cand[0]
            rx_dfe = rx_ffe_dfe_cand[1]
            phase  = ii
        # print(ii, rx_ffe_dfe_cand[-1], phase)
    print(err, phase)
    # --------------------------------------------------------------------------------------------------------------
    # Passing through the Rx FFE
    # --------------------------------------------------------------------------------------------------------------
    rx_ffe_ups   = cdsp.upsample(rx_ffe, osr)
    rx_slicer_in = cdsp.rx.ffe_dfe(ctle_out, rx_ffe_ups, rx_dfe,levels=cdsp.get_levels(constellation, full_scale=full_scale), osr=osr, phase=phase)
    # rx_slicer_in = cdsp.rx.ffe_dfe(ctle_out, rx_ffe_ups, np.array([0]),levels=cdsp.get_levels(constellation, full_scale=full_scale), osr=osr, phase=phase)
    rx_slicer_in = rx_slicer_in[:-1*(len(rx_slicer_in) % osr)] if len(rx_slicer_in) % osr != 0 else rx_slicer_in

    rx_slicer_in_osr1 = cdsp.rx.ffe_dfe(ctle_out_mat.T[phase], rx_ffe, rx_dfe,levels=cdsp.get_levels(constellation, full_scale=full_scale))

    if rx_slicer_in_eye:
        eye_d, amp_vec = cdsp.eye_diagram(rx_slicer_in, osr, 128, fs_value=3, quantization=1024, logscale=False)
        time_ui = np.linspace(0, 2, 256)
        plt.contourf(time_ui, amp_vec, eye_d, levels=100, cmap=cdsp.EYE_COLORMAP)
        plt.title(f'Eye Diagram,  loaded channel + pulse with SNR of {snr} [dB] ; after Rx FFE ')
        plt.xlabel('Time [UI]')
        plt.ylabel('Amplitude')
        plt.show()

    # rx_slicer_in_mat = cdsp.buffer(rx_slicer_in, osr, 0)
    # return rx_slicer_in_mat[:, phase]
    return rx_slicer_in_osr1

def rx_genie_checker():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    prbs_type       = cdsp.constants.PrbsEnum.PRBS13
    constellation   = cdsp.constants.ConstellationEnum.PAM4
    gray_coding     = False
    full_scale      = True
    rx_ffe_out      = rx_example2(rx_slicer_in_eye=True)
    bits_per_symbol = 2
    bit_order_inv   = False
    inv_msb         = False
    inv_lsb         = False
    pn_inv          = False
    # ==================================================================================================================
    # Slicing Rx FFE out to constellation points
    # ==================================================================================================================
    slicer_out = cdsp.rx.slicer(rx_ffe_out, levels=cdsp.get_levels(constellation, full_scale))
    # ==================================================================================================================
    # Decoding
    # ==================================================================================================================
    decoded_dut = cdsp.rx.demapping(slicer_out, constellation, full_scale) if not gray_coding else cdsp.rx.decoding_gray(cdsp.rx.demapping(slicer_out, constellation, full_scale), constellation)
    # ==================================================================================================================
    # Converting to binary
    # ==================================================================================================================
    bit_vec_dut = cdsp.rx.symbol2bin(decoded_dut, 2 ** bits_per_symbol, bit_order_inv, inv_msb, inv_lsb, pn_inv)
    # ==================================================================================================================
    # Checking for errors
    # ==================================================================================================================
    lost_lock, correct_bit_count, error_bit = cdsp.rx.prbs_checker(prbs_type, bit_vec_dut, init_lock=False)
    print(f'Lost lock: {lost_lock}')
    print(f'Correct bit count: {correct_bit_count}')
    print(f'Erred bits: {sum(error_bit)}')


if __name__ == '__main__':
    np.random.seed(140993)
    # tx_example()
    # channel_example(awgn_ch_eye=False)
    # rx_example(ch_out_eye=False, show_ctle=False, ctle_out_eye=False, rx_ffe_eye=True)
    rx_genie_checker()
    # rx_example2(ch_out_eye=False, show_ctle=False, ctle_out_eye=False, rx_slicer_in_eye=True)
