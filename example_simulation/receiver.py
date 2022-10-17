import numpy as np
import CommDspy as cdsp

class Ctle:
    def __init__(self, ctle_data):
        self.zeros   = ctle_data.zeros
        self.poles   = ctle_data.poles
        self.dc_gain = ctle_data.dc_gain
        self.fs      = ctle_data.fs       # sampling frequency
        self.osr     = ctle_data.osr      # Over Sampling
        if ctle_data.zi is None:
            self.zi = np.zeros(max([len(self.zeros), len(self.poles)]))
        else:
            self.zi = ctle_data.zi

    def __call__(self, signal_chunk):
        ctle_out, self.zi = cdsp.rx.ctle(signal_chunk, self.zeros, self.poles, self.dc_gain,
                                         fs=self.fs,
                                         osr=self.osr,
                                         zi=self.zi)
        return ctle_out

class Adc:
    def __init__(self, adc_data):
        self.total_bits  = adc_data.total_bits
        self.frac_bits   = adc_data.frac_bits
        self.quant_type  = adc_data.quant_type
        self.osr         = adc_data.osr
        self.phase       = adc_data.phase
        self.sample_rate = adc_data.sample_rate

    def __call__(self, signal_chunk):
        sampled_signal = signal_chunk[np.arange(self.phase, len(signal_chunk), self.osr//self.sample_rate)]
        adc_out        = cdsp.rx.quantize(sampled_signal, self.total_bits, self.frac_bits, self.quant_type)
        return adc_out

class FfeDfe:
    def __init__(self, ffe_dfe_data):
        self.ffe_precursors  = ffe_dfe_data.ffe_precursors
        self.ffe_postcursors = ffe_dfe_data.ffe_postcursors
        self.dfe_taps        = ffe_dfe_data.dfe_taps
        self.ffe_vec         = ffe_dfe_data.ffe_vec
        self.dfe_vec         = ffe_dfe_data.dfe_vec
        self.levels          = ffe_dfe_data.levels

    def set_ffe(self, ffe_vec):
        if len(ffe_vec) == len(self.ffe_vec):
            self.ffe_vec = ffe_vec
        else:
            raise ValueError('FFE vec length does not match parameters')

    def set_dfe(self,dfe_vec):
        if len(dfe_vec) == self.dfe_taps:
            self.dfe_vec = dfe_vec
        else:
            raise ValueError('DFE vec length does not match parameters')

    def __call__(self, signal_chunk):
        dfe_vec = None if self.dfe_taps == 0 else self.dfe_vec
        slicer_in_vec = cdsp.rx.ffe_dfe(signal_chunk, ffe_taps=self.ffe_vec, dfe_taps=dfe_vec,
                                        levels=self.levels)
        return slicer_in_vec









