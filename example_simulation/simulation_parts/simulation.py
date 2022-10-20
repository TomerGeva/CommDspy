from example_simulation.simulation_parts.transmitter import Transmitter
from example_simulation.simulation_parts.channel import Channel
from example_simulation.simulation_parts.receiver import Receiver

class Simulation:
    def __init__(self, simulation_data, verbose=False):
        self.Tx = Transmitter(simulation_data.prbs_data,
                              simulation_data.coding_data,
                              simulation_data.mapping_data)
        self.Ch = Channel(simulation_data.channel_data)
        self.Rx = Receiver(simulation_data.ctle_data,
                           simulation_data.adc_data,
                           simulation_data.ffe_dfe_data,
                           simulation_data.mapping_data,
                           simulation_data.coding_data)
        self.chunk        = None
        self.ch_out       = None
        self.decoded_bits = None

        self.verbose = verbose

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