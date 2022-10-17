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
    def __init__(self, pulse, pulse_span, ch_type, fir_coefs=None, iir_coefs=None, rolloff=0.35, osr=1, snr=22, pulse_rj_sigma=0, pulse_zi=None, ch_zi=None):
        self.pulse          = pulse
        self.pulse_span     = pulse_span
        self.rolloff        = rolloff
        self.ch_type        = ch_type
        self.iir_coefs      = [1] if iir_coefs is None else fir_coefs
        self.fir_coefs      = [1] if fir_coefs is None else fir_coefs
        self.osr            = osr
        self.snr            = snr
        self.ch_zi          = ch_zi
        self.pulse_zi       = pulse_zi
        self.pulse_rj_sigma = pulse_rj_sigma

class CtleData:
    def __init__(self, zeros, poles, dc_gain, fs, osr, zi=None):
        self.zeros   = zeros
        self.poles   = poles
        self.dc_gain = dc_gain
        self.fs      = fs       # sampling frequency
        self.osr     = osr      # Over Sampling
        self.zi      = zi

class AdcData:
    def __init__(self, total_bits, frac_bits, quant_type):
        self.total_bits = total_bits
        self.frac_bits  = frac_bits
        self.quant_type = quant_type

class FfeDfeData:
    def __init__(self, ffe_precursors, ffe_postcursors, dfe_taps,
                 ffe_vec=None, dfe_vec=None,
                 levels=None):
        self.ffe_precursors  = ffe_precursors
        self.ffe_postcursors = ffe_postcursors
        self.dfe_taps        = dfe_taps
        if ffe_vec is None:
            self.ffe_vec = np.zeros(ffe_precursors+1+ffe_postcursors)
        elif len(ffe_vec) == ffe_precursors+ffe_postcursors+1:
            self.ffe_vec = ffe_vec
        else:
            raise ValueError('FFE vec length does not match parameters')
        if dfe_vec is None:
            self.dfe_vec = np.zeros(dfe_taps)
        elif len(dfe_vec) == dfe_taps:
            self.dfe_vec = dfe_vec
        else:
            raise ValueError('DFE vec length does not match parameters')
        if len(self.dfe_vec) > 0 and levels is None:
            raise ValueError('DFE is needed, but constellation levels are not provided')
        else:
            self.levels = levels