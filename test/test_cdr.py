import numpy as np
import CommDspy as cdsp
from test.auxiliary import generate_pattern, generate_and_pass_channel


def mueller_muller_step_test(prbs_type):
    osr = 64
    ref_pattern, constellation, gray_coding             = generate_pattern(prbs_type)
    channel_ref, channel_out, constellation, assert_str = generate_and_pass_channel(ref_pattern, prbs_type, constellation, gray_coding)
    channel_out, _ = cdsp.channel.pulse_shape(channel_out, osr=osr, span=8, pulse='rcos', beta=0.5)
    channel_out    = channel_out[osr*16:]
    adc_out        = cdsp.rx.quantize(channel_out, 9, 7, 'ss')
    # ==================================================================================================================
    # Estimating optimal phase
    # ==================================================================================================================
    adc_out_mat = cdsp.buffer(adc_out, osr, 0)
    err         = float('inf')
    phase_ref   = -1
    for ii, sampled_phase_data in enumerate(adc_out_mat.T):
        rx_ffe_dfe_cand = cdsp.equalization_estimation_prbs(prbs_type, sampled_phase_data, constellation,
                                                            prbs_full_scale=False,
                                                            ffe_postcursor=len(channel_ref[1:]),
                                                            ffe_precursor=0,
                                                            dfe_taps=0,
                                                            normalize=False,
                                                            bit_order_inv=False,
                                                            pn_inv_precoding=False,
                                                            gray_coded=False,
                                                            pn_inv_postmapping=False)
        if rx_ffe_dfe_cand[-1] < err:
            err       = rx_ffe_dfe_cand[-1]
            phase_ref = ii
    print(f'Best phase is {phase_ref} and the NMSE for this phase is {err:.2f} [dB]')
    # ==================================================================================================================
    # Running Mueller Muller
    # ==================================================================================================================
    prbs_len = 2 ** prbs_type.value - 1
    phase    = int(osr // 2)
    for ii in range(osr):
        signale_phase_signal = adc_out_mat.T[phase]
        pattern_aligned, _   = cdsp.rx.lock_pattern_to_signal(ref_pattern[:prbs_len], signale_phase_signal)
        pattern_aligned_rep  = np.tile(pattern_aligned, int(np.ceil(len(ref_pattern) / prbs_len)))[:len(signale_phase_signal)]
        step_dir = cdsp.rx.mueller_muller_step(signale_phase_signal, pattern_aligned_rep, tol=1e-3)
        phase = (phase + step_dir) % osr
        if phase == 0:
            break
    print(f'Mueller Muller phase is {phase_ref}')
    ch_est, _ = cdsp.channel_estimation_prbs(prbs_type, signale_phase_signal, constellation, channel_postcursor=64)
    assert np.isclose(ch_est[18], ch_est[20], atol=1e-1)