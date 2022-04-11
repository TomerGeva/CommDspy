import CommDspy as cdsp
import numpy as np
import matplotlib.pyplot as plt

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
    coding          = cdsp.constants.CodingEnum.UNCODED
    rolloff         = 0.5
    poly_coeff      = cdsp.get_polynomial(prbs_type)
    init_seed       = np.array([1] * prbs_type.value)
    prbs_len        = 8191  # can be any number
    # ==================================================================================================================
    # Creating reference pattern
    # ==================================================================================================================
    # --------------------------------------------------------------------------------------------------------------
    # Getting PRBS binary pattern
    # --------------------------------------------------------------------------------------------------------------
    prbs_seq, seed_dut = cdsp.tx.prbs_gen(poly_coeff, init_seed, prbs_len)
    # --------------------------------------------------------------------------------------------------------------
    # Duplicating if needed and coding
    # --------------------------------------------------------------------------------------------------------------
    prbs_bin_mult = np.tile(prbs_seq, bits_per_symbol)
    pattern       = cdsp.tx.bin2symbol(prbs_bin_mult, 2 ** bits_per_symbol, bit_order_inv, inv_msb, inv_lsb, pn_inv)
    pattern       = cdsp.tx.coding(pattern, constellation, coding, full_scale=full_scale)
    # ==================================================================================================================
    # Pulse shaping
    # ==================================================================================================================
    tx_out_rect = cdsp.tx.pulse_shape(pattern,osr=32, span=8, method='rect')
    tx_out_sinc = cdsp.tx.pulse_shape(pattern,osr=32, span=8, method='sinc')
    tx_out_rcos = cdsp.tx.pulse_shape(pattern,osr=32, span=8, method='rcos', beta=0.8)
    tx_out_rrc  = cdsp.tx.pulse_shape(pattern,osr=32, span=8, method='rrc', beta=rolloff)
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
    # ==================================================================================================================
    # Eye diagram
    # ==================================================================================================================
    plt.figure()
    eye_d, amp_vec = cdsp.eye_diagram(tx_out_rcos, 32, 128, fs_value=3, quantization=2048, logscale=True)
    time_ui = np.linspace(0, 2, 256)
    plt.contourf(time_ui, amp_vec, eye_d, levels=1000, cmap='gray')
    plt.xlabel('Time [UI]')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    tx_example()
