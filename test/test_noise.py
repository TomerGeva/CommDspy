import numpy as np
import os
import random
from CommDspy.constants import ConstellationEnum, CodingEnum
from CommDspy.noise import awgn
from CommDspy import code_pattern, get_constellation
from test.auxiliary import read_1line_csv


def awgn_test(prbs_type, constellation, snr_db=None):
    """
    :param prbs_type:
    :param constellation:
    :param snr_db:
    :return: Testing the AWGN function.An unbiased estimator of the power variance is given by
                                                    --- N-1
                                              1    \\                    N--> inf
                    VAR(tilde(sigma^2))  = -----   >       (x[n])^2  ------------> sigma^2
                                             N-1   //
                                                   --- n=0
            When checking we need to find the variance of the added noise's power estimation:

                                        2sigma^4
            VAR(tilde(sigma^2)) =   ----------------
                                          N - 1
            We want to test in the 3 std region
    """
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    prbs_len     = 2 ** prbs_type.value - 1
    ref_filename = os.path.join(os.getcwd(), 'test_data', prbs_type.name + '_seed_ones.csv')
    ref_prbs_bin = read_1line_csv(ref_filename)
    snr_db       = random.randint(0, 300) / 10 if snr_db is None else snr_db
    snr_lin      = 10 ** (snr_db / 10)
    assert_str   = '|{0:^8s}|{1:^6s}|'.format(prbs_type.name, constellation.name)
    # ==================================================================================================================
    # Manipulating prbs based on the constellation and coding
    # ==================================================================================================================
    if constellation == ConstellationEnum.PAM4:
        ref_pattern = np.reshape(np.tile(ref_prbs_bin, 2), [-1, 2]).dot(np.array([1, 2]))
    else:
        ref_pattern = ref_prbs_bin
    ref_pattern = code_pattern(ref_pattern, constellation, coding=CodingEnum.UNCODED, pn_inv=False)
    # ==================================================================================================================
    # Calling DUT
    # ==================================================================================================================
    noisey_pattern  = awgn(ref_pattern, snr_db)
    noise           = noisey_pattern - ref_pattern
    noise_power_hat = np.var(noise, ddof=1)
    # ==================================================================================================================
    # Checking
    # ==================================================================================================================
    ref_signal_power = np.mean(get_constellation(constellation) ** 2)
    noise_power_ref  = ref_signal_power / snr_lin
    noise_power_hat_var = 2 * noise_power_ref ** 2 / (prbs_len - 1)
    assert np.abs(noise_power_hat - noise_power_ref) < 3 * np.sqrt(noise_power_hat_var),assert_str + ' AWGN SNR test Failed !!!'

