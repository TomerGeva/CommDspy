import os
import json
from constants import *
from example_simulation.simulation_parts.data_objects import PrbsData, CodingData, MappingData, ChannelData, CtleData, AdcData, FfeDfeData, FullLinkData
# ======================================================================================================================
# Tx data
# ======================================================================================================================
Prbs_data = PrbsData(PRBS_TYPE,
                     chunk_size=CHUNK_SIZE,
                     init_seed=None)
# ------------------------------------------------------------------------------------------------------------------
# Coding + Mapping data
# ------------------------------------------------------------------------------------------------------------------
Coding_data  = CodingData(BITS_PER_SYMBOL, CONSTELLATION, CHUNK_SIZE,
                          coding_conv=True, G=G, feedback=FEEDBACK, use_feedback=USE_FEEDBACK,
                          inv_msb=False,
                          inv_lsb=False,
                          pn_inv=False,
                          coding_gray=True)
Mapping_data = MappingData(CONSTELLATION, LEVELS, AMP_PP_MV, RX_FACTOR)
# ======================================================================================================================
# Channel data
# ======================================================================================================================
f    = open(os.path.join('../test/example_scripts', '..', 'test_data', 'example_channel_full.json'))
data = json.load(f)
f.close()
channel_isi = cdsp.upsample(data['channel_sampled'], OSR)
# channel_isi = cdsp.upsample(np.array([0,0.02,0.15,0.04]), OSR)
del f, data
Channel_data = ChannelData(PULSE, PULSE_SPAN, CHANNEL_TYPE,
                           fir_coefs=channel_isi,
                           rolloff=ROLLOFF,
                           pulse_rj_sigma=RJ_SIGMA,
                           osr=OSR,
                           snr=SNR)
# ======================================================================================================================
# Receiver data
# ======================================================================================================================
Ctle_data    = CtleData(ZEROS, POLES, DC_GAIN, FS, osr=OSR)
Adc_data     = AdcData(ADC_BITS, FRAC_BITS, QUANT_TYPE, osr=OSR, sample_rate=SAMPLE_RATE)
Ffe_dfe_data = FfeDfeData(FFE_PRECURSORS, FFE_POSTCURSORS, DFE_TAPS,
                          ffe_vec=FFE_VEC, dfe_vec=DFE_VEC, levels=SLICER_LEVELS)

Link_data = FullLinkData(Prbs_data, Coding_data, Mapping_data, Channel_data, Ctle_data, Adc_data, Ffe_dfe_data)
# ======================================================================================================================
# Control data
# ======================================================================================================================
Control_vars = ControlVars()
