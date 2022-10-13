import numpy as np
import CommDspy as cdsp
from data_objects import PrbsData, CodingData
from transmitter import Transmitter
from channel import Channel
from config_file import *

def main():
    Tx = Transmitter(Prbs_data, Coding_data, Mapping_data)
    Ch = Channel(Channel_data)

    chunk  = Tx.generate()
    ch_out = Ch.pass_through(chunk)
    print('hi')


if __name__ == '__main__':
    main()