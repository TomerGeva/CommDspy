from transmitter import Transmitter
from channel import Channel
from config_file import *

def main():
    Tx = Transmitter(Prbs_data, Coding_data, Mapping_data)
    Ch = Channel(Channel_data)

    print('Generating')
    chunk  = Tx.generate()
    print('Passing')
    ch_out = Ch(chunk)
    print('hi')


if __name__ == '__main__':
    main()