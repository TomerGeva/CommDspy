from example_simulation.simulation_parts.transmitter import Transmitter
from example_simulation.simulation_parts.channel import Channel
from example_simulation.simulation_parts.receiver import Receiver

class FullLink:
    def __init__(self, link_data, verbose=False):
        self.Tx = Transmitter(link_data.prbs_data,
                              link_data.coding_data,
                              link_data.mapping_data)
        self.Ch = Channel(link_data.channel_data)
        self.Rx = Receiver(link_data.ctle_data,
                           link_data.adc_data,
                           link_data.ffe_dfe_data,
                           link_data.mapping_data,
                           link_data.coding_data)
        self.chunk        = None
        self.ch_out       = None
        self.decoded_bits = None

        self.verbose = verbose

    def start_convergence(self):
        self.Rx.converge     = True
        self.Rx.converge_lms = True
        self.Rx.converge_cdr = True
        self.Rx.lms_mse_vec  = []
        self.Rx.cdr_step_vec = []

    def __call__(self):
        if self.verbose:
            print('Generating chunk')
        self.chunk  = self.Tx.generate()
        if self.verbose:
            print('Passing chunk through the channel')
        self.ch_out = self.Ch(self.chunk)
        if self.verbose:
            print('Passing chunk through the Rx')
        self.decoded_bits = self.Rx(self.ch_out)
        return self.decoded_bits
