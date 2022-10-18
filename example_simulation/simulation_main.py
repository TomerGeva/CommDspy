from transmitter import Transmitter
from channel import Channel
from receiver import Receiver
from config_file import *

def main():
    np.random.seed(140993)
    Tx = Transmitter(Prbs_data, Coding_data, Mapping_data)
    Ch = Channel(Channel_data)
    Rx = Receiver(Ctle_data, Adc_data, Ffe_dfe_data, Mapping_data, Coding_data)
    for ii in range(2):
        print('Generating')
        chunk  = Tx.generate()
        print('Passing channel')
        ch_out = Ch(chunk)
        print('Passing Rx')
        received_data = Rx(ch_out)
    print('hi')


if __name__ == '__main__':
    main()