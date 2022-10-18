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
        adc_out        = cdsp.rx.quantize(sampled_signal, self.total_bits, 0, self.quant_type)
        return adc_out / (2**self.frac_bits)

class FfeDfe:
    def __init__(self, ffe_dfe_data):
        self.ffe_precursors  = ffe_dfe_data.ffe_precursors
        self.ffe_postcursors = ffe_dfe_data.ffe_postcursors
        self.dfe_taps        = ffe_dfe_data.dfe_taps
        self.ffe_vec         = ffe_dfe_data.ffe_vec
        self.dfe_vec         = ffe_dfe_data.dfe_vec
        self.levels          = ffe_dfe_data.levels
        self.zi_ffe          = ffe_dfe_data.zi_ffe
        self.zi_dfe          = ffe_dfe_data.zi_dfe

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
        if len(self.ffe_vec) == 1 and self.dfe_taps == 0:  # no FFE and no DFE
            slicer_in_vec = cdsp.rx.ffe_dfe(signal_chunk, ffe_taps=self.ffe_vec, dfe_taps=dfe_vec,
                                            levels=self.levels)
        elif len(self.ffe_vec) == 1:  # only DFE
            slicer_in_vec, self.zi_dfe = cdsp.rx.ffe_dfe(signal_chunk,
                                                         ffe_taps=self.ffe_vec, dfe_taps=dfe_vec,
                                                         levels=self.levels,
                                                         zi_dfe=self.zi_dfe)
        elif self.dfe_taps == 0:  # only FFE
            slicer_in_vec, self.zi_ffe = cdsp.rx.ffe_dfe(signal_chunk,
                                                         ffe_taps=self.ffe_vec, dfe_taps=dfe_vec,
                                                         levels=self.levels,
                                                         zi_ffe=self.zi_ffe)
        else:  # Both FFE and DFE
            slicer_in_vec, self.zi_ffe, self.zi_dfe = cdsp.rx.ffe_dfe(signal_chunk,
                                                                      ffe_taps=self.ffe_vec, dfe_taps=dfe_vec,
                                                                      levels=self.levels,
                                                                      zi_ffe=self.zi_ffe, zi_dfe=self.zi_dfe)
        return slicer_in_vec

class Decoder:
    def __init__(self, coding_data):
        self.constellation           = coding_data.constellation
        self.bits_per_symbol         = coding_data.bits_per_symbol
        self.chunk_size              = coding_data.chunk_size
        self.bit_order_inv           = coding_data.bit_order_inv
        self.inv_msb                 = coding_data.inv_msb
        self.inv_lsb                 = coding_data.inv_lsb
        self.pn_inv                  = coding_data.pn_inv
        self.coding_gray             = coding_data.coding_gray
        self.coding_differential     = coding_data.coding_differential
        self.coding_bin_differential = coding_data.coding_bin_differential
        self.coding_manchester       = coding_data.coding_manchester
        self.coding_bipolar          = coding_data.coding_bipolar
        self.coding_mlt3             = coding_data.coding_mlt3
        self.coding_diff_manchester  = coding_data.coding_diff_manchester
        self.coding_linear           = coding_data.coding_linear
        self.coding_conv             = coding_data.coding_conv
        self.G                       = coding_data.G
        self.feedback                = coding_data.feedback
        self.use_feedback            = coding_data.use_feedback

    def __call__(self, signal_chunk):
        # ==============================================================================================================
        # Symbol encoding
        # ==============================================================================================================
        pattern_decoded = signal_chunk.copy()
        if self.coding_gray:
            pattern_decoded = cdsp.rx.decoding_differential(pattern_decoded, self.constellation)
        if self.coding_differential:
            pattern_decoded = cdsp.rx.decoding_gray(pattern_decoded, self.constellation)
        # ==============================================================================================================
        # Converting symbols to binary if needed
        # ==============================================================================================================
        if self.bits_per_symbol > 1:
            pattern_decoded = cdsp.rx.symbol2bin(pattern_decoded,
                                                 num_of_symbols=2**self.bits_per_symbol,
                                                 bit_order_inv=self.bit_order_inv,
                                                 inv_msb=self.inv_msb,
                                                 inv_lsb=self.inv_lsb,
                                                 pn_inv=self.pn_inv)
        elif np.isclose(int(self.bits_per_symbol), self.bits_per_symbol):
        # meaning two levels, only pn_inv is relevant here
            if self.pn_inv:
                pattern_decoded = 1 - pattern_decoded
        else:
        # meaning three levels, at this point it is {0, 1, 2}
            if self.pn_inv:
                pattern_decoded = 2 - pattern_decoded
        # ==============================================================================================================
        # Binary decoding options
        # ==============================================================================================================
        if self.coding_bin_differential:
            pattern_decoded = cdsp.rx.decoding_differential(pattern_decoded, cdsp.ConstellationEnum.OOK)
        if self.coding_manchester:
            pattern_decoded = cdsp.rx.decoding_manchester(pattern_decoded)
        if self.coding_bipolar:
            pattern_decoded = cdsp.rx.decoding_bipolar(pattern_decoded)
        if self.coding_mlt3:
            pattern_decoded = cdsp.rx.decoding_mlt3(pattern_decoded)
        if self.coding_diff_manchester:
            pattern_decoded = cdsp.rx.decoding_differential_manchester(pattern_decoded)
        if self.coding_linear:
            pattern_decoded = cdsp.rx.decoding_linear(pattern_decoded, self.G)
        if self.coding_conv:
            pattern_decoded, _ = cdsp.rx.decoding_conv_viterbi(pattern_decoded, self.G,
                                                               tb_len=self.chunk_size,
                                                               feedback=self.feedback,
                                                               use_feedback=self.use_feedback)
        return pattern_decoded
class Receiver:
    def __init__(self, ctle_data, adc_data, ffe_dfe_data, mapping_data, coding_data):
        self.ctle    = Ctle(ctle_data)
        self.adc     = Adc(adc_data)
        self.ffe_dfe = FfeDfe(ffe_dfe_data)
        self.constellation = mapping_data.constellation
        self.levels        = (mapping_data.levels * 2 / mapping_data.amp_pp_mv) * mapping_data.rx_factor
        self.decoder = Decoder(coding_data)
        # ==============================================================================================================
        # Memory
        # ==============================================================================================================
        self.ctle_out       = np.array([0])
        self.adc_out        = np.array([0])
        self.slicer_in      = np.array([0])
        self.slicer_out     = np.array([0])
        self.demapped_chunk = np.array([0])
        self.decoded_chunk  = np.array([0])

    def input_stage(self, continuous_chunk):
        self.ctle_out = self.ctle(continuous_chunk)
        self.adc_out  = self.adc(self.ctle_out)

    def digital_stage(self):
        self.slicer_in  = self.ffe_dfe(self.adc_out)
        self.slicer_out = cdsp.rx.slicer(self.slicer_in, self.levels)

    def extract_data(self):
        self.demapped_chunk = cdsp.rx.demapping(self.slicer_out, self.constellation)
        self.decoded_chunk  = self.decoder(self.demapped_chunk)
        return self.decoded_chunk

    def __call__(self, continuous_chunk):
        self.input_stage(continuous_chunk)
        self.digital_stage()
        received_data = self.extract_data()
        return received_data







