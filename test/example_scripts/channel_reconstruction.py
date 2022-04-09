import numpy as np
import os
import CommDspy as cdsp
import json
import matplotlib.pyplot as plt

def channel_reconstruction():
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    order = 16
    beta  = 0.5
    # ==================================================================================================================
    # Loading data
    # ==================================================================================================================
    f = open(os.path.join('..','test_data', 'example_channel_full.json'))
    data = json.load(f)
    f.close()
    channel             = np.array(data['channel'])
    channel_ui          = data['channel_ui']
    channel_sampled     = data['channel_sampled']
    osr                 = data['osr']
    ch_segment          = channel[order // 2 * osr: -1*(((order // 2)-1) * osr - 1)]
    # ==================================================================================================================
    # Digital up-sampling DUT
    # ==================================================================================================================
    channel_upsample_lag, x2, x1 = cdsp.digital_oversample(channel_sampled, osr=osr, order=order, method='lagrange')
    channel_upsample_sinc, _, _  = cdsp.digital_oversample(channel_sampled, osr=osr, order=order, method='sinc')
    channel_upsample_rcos, _, _  = cdsp.digital_oversample(channel_sampled, osr=osr, order=order, method='rcos', beta=beta)
    # ==================================================================================================================
    # Plotting
    # ==================================================================================================================
    plt.plot([ch_idx - (order // 2) for ch_idx in channel_ui], channel)
    plt.plot(x2, channel_upsample_sinc)
    plt.plot(x2, channel_upsample_lag)
    plt.plot(x2, channel_upsample_rcos)
    plt.plot(x1, channel_sampled[(order//2):-1*(order//2)+1], 'o')
    plt.grid()
    plt.legend(['original channel',
                'reconstructed channel - sinc',
                'reconstructed channel - Lagrange',
                f'reconstructed channel - rcos (beta = {beta})',
                'sampled channel'])
    plt.title('Digital oversampling of an example channel')
    plt.xlim([0, 20])
    print(f'Sinc interpolation MSE          = {10*np.log10(cdsp.power(ch_segment - channel_upsample_sinc) / cdsp.power(ch_segment)):#.2f} [dB]')
    print(f'Lagrange interpolation MSE      = {10*np.log10(cdsp.power(ch_segment - channel_upsample_lag)  / cdsp.power(ch_segment)):#.2f} [dB]')
    print(f'Raised Cosine interpolation MSE = {10*np.log10(cdsp.power(ch_segment - channel_upsample_rcos) / cdsp.power(ch_segment)):#.2f} [dB]')
    plt.show()

if __name__ == '__main__':
    channel_reconstruction()