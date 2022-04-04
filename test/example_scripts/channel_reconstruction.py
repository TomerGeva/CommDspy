import numpy as np
import random
import csv
import os
from CommDspy.constants import PrbsEnum, ConstellationEnum, CodingEnum
import CommDspy as cdsp
import json

def dig_over_sample_channel():
    import matplotlib.pyplot as plt
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    osr   = 16
    order = 16
    # ==================================================================================================================
    # Loading data
    # ==================================================================================================================
    f = open(os.path.join('..','test_data', 'example_channel_full.json'))
    data = json.load(f)
    f.close()
    channel             = data['channel']
    channel_ui          = data['channel_ui']
    channel_sampled     = data['channel_sampled']
    # ==================================================================================================================
    # Digital up-sampling DUT
    # ==================================================================================================================
    channel_upsample_lag, x2, x1 = cdsp.digital_oversample(channel_sampled, osr=osr, order=order, method='lagrange')
    channel_upsample_sinc, _, _  = cdsp.digital_oversample(channel_sampled, osr=osr, order=order, method='sinc')
    # ==================================================================================================================
    # Plotting
    # ==================================================================================================================
    plt.plot([ch_idx - (order // 2) for ch_idx in channel_ui], channel)
    plt.plot(x2, channel_upsample_lag)
    plt.plot(x2, channel_upsample_sinc)
    plt.plot(x1, channel_sampled[(order//2):-1*(order//2)+1], 'o')
    plt.grid()
    plt.legend(['original channel', 'reconstructed channel - Lagrange', 'reconstructed channel - sinc', 'sampled channel'])
    plt.xlim([0, 20])
    plt.show()

if __name__ == '__main__':
    dig_over_sample_channel()