import numpy as np
import CommDspy as cdsp
import os
import json
from example_simulation.data_objects import PrbsData, CodingData, MappingData, ChannelData
# ======================================================================================================================
# Tx data
# ======================================================================================================================
# ------------------------------------------------------------------------------------------------------------------
# PRBS generator
# ------------------------------------------------------------------------------------------------------------------
Prbs_data = PrbsData(cdsp.constants.PrbsEnum.PRBS13,
                     chunk_size=64,
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
Coding_data  = CodingData(bits_per_symbol, constellation,
                          coding_conv=True, G=G, feedback=feedback, use_feedback=use_feedback,
                          inv_msb=False,
                          inv_lsb=False,
                          pn_inv=False,
                          coding_gray=True)
amp_pp_mv    = 750  # transmition amplitude peak to peak in [mV]
Mapping_data = MappingData(constellation, levels, amp_pp_mv)
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
