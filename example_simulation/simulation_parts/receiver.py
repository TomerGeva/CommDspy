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
        self.levels          = ffe_dfe_data.levels
        # ==============================================================================================================
        # Memory
        # ==============================================================================================================
        self.ffe_vec         = ffe_dfe_data.ffe_vec
        self.dfe_vec         = ffe_dfe_data.dfe_vec
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
    def __init__(self, ctle_data, adc_data, ffe_dfe_data, mapping_data, coding_data, lms_lr=2**(-10)):
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
        self.ctle_out_last       = np.array([0])
        self.adc_out_last        = np.array([0])
        self.slicer_in_last      = np.array([0])
        self.slicer_out_last     = np.array([0])
        self.demapped_chunk_last = np.array([0])
        self.decoded_chunk_last  = np.array([0])

        self.lms_mse_last = 1e3
        # ==============================================================================================================
        # Control bits
        # ==============================================================================================================
        self.converge        = False
        self.converge_lms    = False
        self.converge_cdr    = False
        self.converge_done   = False
        self.lms_ffe_idx     = np.arange(-1 * self.ffe_dfe.ffe_precursors, self.ffe_dfe.ffe_postcursors+1)
        self.lms_dfe_idx     = np.arange(1, self.ffe_dfe.dfe_taps+1)
        self.lms_lr          = lms_lr
        self.lms_mse_diff_th = -40  # [dB]
        self.phase           = adc_data.phase
        self.cdr_step_vec    = []
        self.max_pase        = ctle_data.osr

    def input_stage(self, continuous_chunk):
        self.ctle_out_last = self.ctle_out
        self.adc_out_last  = self.adc_out
        self.ctle_out = self.ctle(continuous_chunk)
        self.adc_out  = self.adc(self.ctle_out)

    def digital_stage(self):
        self.slicer_in_last  = self.slicer_in
        self.slicer_out_last = self.slicer_out
        self.slicer_in  = self.ffe_dfe(self.adc_out)
        self.slicer_out = cdsp.rx.slicer(self.slicer_in, self.levels)
        if self.converge:
            delay          = self.ffe_dfe.ffe_precursors
            if delay > 0:
                adc_vec    = np.concatenate([self.adc_out_last[-1*delay:], self.adc_out[:-1*delay]])
            else:
                adc_vec    = self.adc_vec
            # adc_vec        = np.concatenate([self.adc_out_last, self.adc_out])
            # slicer_out_vec = np.concatenate([self.slicer_out_last, self.slicer_out])
            # ------------------------------------------------------------------------------------------------------
            # Converging the CDR
            # ------------------------------------------------------------------------------------------------------
            if self.converge_cdr:
                mm_step = cdsp.rx.mueller_muller_step(adc_vec, self.slicer_out)
                # if delay > 0:
                #     mm_step = cdsp.rx.mueller_muller_step(adc_vec[:-1*delay], slicer_out_vec[delay:])
                # else:
                #     mm_step = cdsp.rx.mueller_muller_step(adc_vec, slicer_out_vec)
                self.phase     += mm_step
                self.adc.phase += mm_step
                self.cdr_step_vec.append(mm_step)
                # **********************************************************************************************
                # Stop condition
                # **********************************************************************************************
                if mm_step == 0:
                    self.converge_cdr = False
            # ------------------------------------------------------------------------------------------------------
            # Converging the FFE and DFE
            # ------------------------------------------------------------------------------------------------------
            if self.converge_lms:
                mse, ffe_grad_vec, dfe_grad_vec = cdsp.rx.lms_grad(adc_vec, self.slicer_in, self.levels,
                                                                   ffe_tap_idx=self.lms_ffe_idx,
                                                                   dfe_tap_idx=self.lms_dfe_idx)
                self.ffe_dfe.ffe_vec += -1 * self.lms_lr * ffe_grad_vec
                self.ffe_dfe.dfe_vec += -1 * self.lms_lr * dfe_grad_vec
                self.lms_mse_last = mse
                # **********************************************************************************************
                # Stop condition
                # **********************************************************************************************
                if abs(mse - self.lms_mse_last) > 10 ** (self.lms_mse_diff_th / 20):
                    self.converge_lms = False
            if not self.converge_cdr and not self.converge_lms:
                self.converge      = False
                self.converge_done = True

    def extract_data(self):
        self.demapped_chunk_last = self.demapped_chunk
        self.decoded_chunk_last  = self.decoded_chunk
        self.demapped_chunk = cdsp.rx.demapping(self.slicer_out, self.constellation)
        self.decoded_chunk  = self.decoder(self.demapped_chunk)
        return self.decoded_chunk

    def __call__(self, continuous_chunk):
        self.input_stage(continuous_chunk)
        self.digital_stage()
        received_data = self.extract_data()
        return received_data







