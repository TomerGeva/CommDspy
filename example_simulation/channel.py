import numpy as np
import CommDspy as cdsp

class Channel:
    def __init__(self, channel_data):
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        self.pulse          = channel_data.pulse
        self.pulse_span     = channel_data.pulse_span
        self.rolloff        = channel_data.rolloff
        self.ch_type        = channel_data.ch_type
        self.fir_coefs      = channel_data.fir_coefs
        self.iir_coefs      = channel_data.iir_coefs
        self.osr            = channel_data.osr
        self.snr            = channel_data.snr
        self.pulse_rj_sigma = channel_data.pulse_rj_sigma
        # ==============================================================================================================
        # Memory
        # ==============================================================================================================
        if channel_data.ch_memory is None:
            if type(self.iir_coefs) in [list, np.ndarray]:
                self.ch_memory = np.zeros(max([len(self.iir_coefs), len(self.fir_coefs)])-1)
            else:  # only FIR
                self.ch_memory = np.zeros(max([1, len(self.fir_coefs)])-1)
        else:
            self.ch_memory     = channel_data.ch_memory
        if channel_data.pulse_memory is None:
            pulse_len_fui = self.pulse_span * self.osr * 2  # pulse length is fractional of UI
            self.pulse_memory  = np.zeros(pulse_len_fui)
        else:
            self.pulse_memory  = channel_data.pulse_memory

    def pass_through(self, signal_chunk):
        pulse_out, self.pulse_memory = cdsp.channel.pulse_shape(signal_chunk,
                                                                osr=self.osr,
                                                                span=self.pulse_span,
                                                                pulse=self.pulse,
                                                                beta=self.rolloff,
                                                                rj_sigma=self.pulse_rj_sigma,
                                                                zi=self.pulse_memory)
        if self.ch_type == 'pulse':
            return pulse_out
        elif self.ch_type == 'awgn':
            return cdsp.channel.awgn(pulse_out, snr=self.snr)
        elif self.ch_type == 'isi_awgn':
            ch_out, self.ch_memory = cdsp.channel.awgn_channel(pulse_out, self.fir_coefs, self.iir_coefs,
                                                               zi=self.ch_memory,
                                                               snr=self.snr)
            return ch_out

