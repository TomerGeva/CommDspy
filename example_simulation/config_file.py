import numpy as np
import CommDspy as cdsp
import os
import json
from example_simulation.data_objects import PrbsData, CodingData, MappingData, ChannelData, CtleData, AdcData, FfeDfeData
# ======================================================================================================================
# Tx data
# ======================================================================================================================
# ------------------------------------------------------------------------------------------------------------------
# PRBS generator
# ------------------------------------------------------------------------------------------------------------------
chunk_size = 64
Prbs_data = PrbsData(cdsp.constants.PrbsEnum.PRBS13,
                     chunk_size=chunk_size,
                     init_seed=None)
# ------------------------------------------------------------------------------------------------------------------
# Coding + Mapping data
# ------------------------------------------------------------------------------------------------------------------
constellation   = cdsp.constants.ConstellationEnum.PAM4
levels          = cdsp.get_levels(constellation, full_scale=False)

bits_per_symbol = np.log2(len(levels))
G            = {0:np.array([[1, 0],[0, 1]])}  # for convolution coding
feedback     = {0:np.array([1, 1])}
use_feedback = np.array([[0, 1]])
Coding_data  = CodingData(bits_per_symbol, constellation, chunk_size,
                          coding_conv=True, G=G, feedback=feedback, use_feedback=use_feedback,
                          inv_msb=False,
                          inv_lsb=False,
                          pn_inv=False,
                          coding_gray=True)
amp_pp_mv    = 750  # transmition amplitude peak to peak in [mV]
rx_factor    = 0.75
Mapping_data = MappingData(constellation, levels, amp_pp_mv, rx_factor)
# ======================================================================================================================
# Channel data
# ======================================================================================================================
snr        = 10     # [dB], AWGN at the moment
pulse      = 'rcos' # raised cosine pulse
rolloff    = 0.35   # pulse rolloff factor
pulse_span = 8      # [UI], how long is the pulse
osr        = 32     # over sampling rate
rj_sigma   = 0      # random jitter std

channel_type = 'isi_awgn'
f    = open(os.path.join('../test/example_scripts', '..', 'test_data', 'example_channel_full.json'))
data = json.load(f)
f.close()
channel_isi = cdsp.upsample(data['channel_sampled'], osr)
del f, data
Channel_data = ChannelData(pulse, pulse_span, channel_type,
                           fir_coefs=channel_isi,
                           rolloff=rolloff,
                           pulse_rj_sigma=rj_sigma,
                           osr=osr,
                           snr=snr)
# ======================================================================================================================
# Receiver data
# ======================================================================================================================
# ------------------------------------------------------------------------------------------------------------------
# CTLE
# ------------------------------------------------------------------------------------------------------------------
zeros     = [5e8, 11e9]
poles     = [1e9, 20e9, 25e9]
dc_gain   = -10  # [dB]
fs        = 53.125e9
Ctle_data = CtleData(zeros, poles, dc_gain, fs, osr=osr)
# ------------------------------------------------------------------------------------------------------------------
# ADC
# ------------------------------------------------------------------------------------------------------------------
total_bits  = 8
frac_bits   = 6
quant_type  = 'ss'
sample_rate = 1  # samples per UI
Adc_data    = AdcData(total_bits, frac_bits, quant_type, osr=osr, sample_rate=sample_rate)
# ------------------------------------------------------------------------------------------------------------------
# FFE DFE
# ------------------------------------------------------------------------------------------------------------------
ffe_precursors  = 4
ffe_postcursors = 27
dfe_taps        = 1
ffe_vec         = np.array([ 5.07514881e-02, -2.49866354e-01,  8.38558833e-01, -2.56477841e+00,
                             6.01132793e+00, -1.36714831e+00, -1.23461720e+00,  5.13458332e-01,
                             4.59859613e-02, -3.45056732e-02,  1.00430478e-01,  2.21703087e-02,
                             1.22324600e-02,  4.83370272e-02, -3.74533635e-03,  3.10001097e-02,
                             3.39484133e-02,  1.75975037e-02,  2.40305033e-02,  1.14154910e-02,
                            -5.14113769e-02,  1.55452941e-01, -1.20958476e-01,  9.69672233e-02,
                             4.67353341e-02, -7.95786424e-02,  4.61269177e-02, -6.89049422e-03,
                             6.00878497e-02, -4.06378296e-02,  6.33841413e-02, -1.79156648e-02])
dfe_vec         = np.array([0.4147687])
slicer_levels   = levels
Ffe_dfe_data    = FfeDfeData(ffe_precursors, ffe_postcursors, dfe_taps,
                             ffe_vec=ffe_vec, dfe_vec=dfe_vec, levels=slicer_levels)
