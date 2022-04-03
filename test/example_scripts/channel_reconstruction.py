import numpy as np
import random
import csv
import os
import test
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
    f = open(os.path.join('test_data', 'example_channel_full.json'))
    data = json.load(f)
    f.close()
    channel             = data['channel']
    channel_ui          = data['channel_ui']
    channel_sampled     = data['channel_sampled']
    # ==================================================================================================================
    # Digital up-sampling DUT
    # ==================================================================================================================
    channel_upsample, x2, x1 = cdsp.digital_oversample(channel_sampled, osr=osr, order=order)
    # ==================================================================================================================
    # Plotting
    # ==================================================================================================================
    plt.plot([ch_idx - (order // 2) for ch_idx in channel_ui], channel)
    plt.plot(x2, channel_upsample)
    plt.plot(x1, channel_sampled[(order//2):-1*(order//2)+1], 'o')
    plt.grid()
    plt.legend(['original channel', 'reconstructed channel', 'sampled channel'])
    plt.xlim([0, 20])
    plt.show()