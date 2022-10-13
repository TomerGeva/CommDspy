import numpy as np
import CommDspy as cdsp

class PrbsData:
    def __init__(self, prbs_type, chunk_size, init_seed=None):
        self.prbs_type  = prbs_type
        self.gen_poly   = cdsp.get_polynomial(prbs_type)
        self.chunk_size = chunk_size
        self.seed       = init_seed if init_seed is not None else np.ones_like(self.gen_poly)

class CodingData:
    def __init__(self, bits_per_symbol, constellation,
                 bit_order_inv=False,
                 inv_msb=False, inv_lsb=False,
                 pn_inv=False,
                 coding_gray=False,
                 coding_differential=False,
                 coding_bin_differential=False,
                 coding_manchester=False,
                 coding_bipolar=False,
                 coding_mlt3=False,
                 coding_diff_manchester=False,
                 coding_linear=False,
                 coding_conv=False,
                 G=None,
                 feedback=None,
                 use_feedback=None):
        self.constellation   = constellation
        self.bits_per_symbol = bits_per_symbol
        self.bit_order_inv   = bit_order_inv
        self.inv_msb         = inv_msb
        self.inv_lsb         = inv_lsb
        self.pn_inv          = pn_inv
        self.coding_gray     = coding_gray
        self.coding_differential     = coding_differential
        self.coding_bin_differential = coding_bin_differential
        self.coding_manchester       = coding_manchester
        self.coding_bipolar          = coding_bipolar
        self.coding_mlt3             = coding_mlt3
        self.coding_diff_manchester  = coding_diff_manchester
        self.coding_linear           = coding_linear
        self.coding_conv             = coding_conv
        self.G            = G
        self.feedback     = feedback
        self.use_feedback = use_feedback

class MappingData:
    def __init__(self, constellation, levels=None, amp_pp_mv=2):
        self.constellation = constellation
        if levels is None:
            self.levels = cdsp.get_levels(constellation)
        else:
            self.levels = levels / np.max(np.abs(levels)) * amp_pp_mv / 2

class ChannelData:
    def __init__(self, pulse, pulse_span, ch_type, fir_coefs=None, iir_coefs=None, rolloff=0.35, osr=1, snr=22, pulse_rj_sigma=0, pulse_memory=None, ch_memory=None):
        self.pulse          = pulse
        self.pulse_span     = pulse_span
        self.rolloff        = rolloff
        self.ch_type        = ch_type
        self.iir_coefs      = [1] if iir_coefs is None else fir_coefs
        self.fir_coefs      = [1] if fir_coefs is None else fir_coefs
        self.osr            = osr
        self.snr            = snr
        self.ch_memory      = ch_memory
        self.pulse_memory   = pulse_memory
        self.pulse_rj_sigma = pulse_rj_sigma



