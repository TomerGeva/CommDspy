from transmitter import Transmitter
from channel import Channel
from receiver import Ctle, Adc
from config_file import *

def main():
    np.random.seed(140993)
    Tx = Transmitter(Prbs_data, Coding_data, Mapping_data)
    Ch = Channel(Channel_data)
    ctle = Ctle(Ctle_data)
    adc  = Adc(Adc_data)
    for ii in range(2):
        print('Generating')
        chunk  = Tx.generate()
        print('Passing channel')
        ch_out = Ch(chunk)
        print('Passing CTLE')
        ctle_out = ctle(ch_out)
        print('Passing ADC')
        adc_out = adc(ctle_out)
    print('hi')



if __name__ == '__main__':
    main()