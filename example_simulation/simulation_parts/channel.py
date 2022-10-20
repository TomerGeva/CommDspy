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
        if channel_data.ch_zi is None:
            if type(self.iir_coefs) in [list, np.ndarray]:
                self.ch_zi = np.zeros(max([len(self.iir_coefs), len(self.fir_coefs)])-1)
            else:  # only FIR
                self.ch_zi = np.zeros(max([1, len(self.fir_coefs)])-1)
        else:
            self.ch_zi     = channel_data.ch_zi
        if channel_data.pulse_zi is None:
            pulse_len_fui = self.pulse_span * self.osr * 2  # pulse length is fractional of UI
            self.pulse_zi = np.zeros(pulse_len_fui)
        else:
            self.pulse_zi  = channel_data.pulse_zi

    def __call__(self, signal_chunk):
        pulse_out, self.pulse_zi = cdsp.channel.pulse_shape(signal_chunk,
                                                                osr=self.osr,
                                                                span=self.pulse_span,
                                                                pulse=self.pulse,
                                                                beta=self.rolloff,
                                                                rj_sigma=self.pulse_rj_sigma,
                                                                zi=self.pulse_zi)
        if self.ch_type == 'pulse':
            return pulse_out
        elif self.ch_type == 'awgn':
            return cdsp.channel.awgn(pulse_out, snr=self.snr)
        elif self.ch_type == 'isi_awgn':
            ch_out, self.ch_zi = cdsp.channel.awgn_channel(pulse_out, self.fir_coefs, self.iir_coefs,
                                                           zi=self.ch_zi,
                                                           snr=self.snr)
            return ch_out

