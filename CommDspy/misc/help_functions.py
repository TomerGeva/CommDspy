import numpy as np

def check_binary(pattern):
    data_in = np.unique(pattern)
    if len(data_in) == 1 and (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other encoding methods')
    elif len(data_in) == 2 and (0 not in data_in and 1 not in data_in):
        raise ValueError('Data in is not binary, please consider other encoding methods')
    elif len(data_in) > 2:
        raise ValueError('Data in is not binary, please consider other encoding methods')

def check_valid_conv(pattern, G, feedback, use_feedback):
    check_binary(pattern)
    for ii in G:
        check_binary(G[ii])
        # ----------------------------------------------------------------------------------------------------------
        # Checking that the feedback rules are OK and that all is binary
        # ----------------------------------------------------------------------------------------------------------
        if feedback is not None:
            if use_feedback is None:
                raise ValueError('use_feedback must not be None is feedback is used')
            elif use_feedback.shape[0] != len(G) or use_feedback.shape[1] != G[0].shape[0]:
                raise ValueError('Size of use_feedback should be similar to the number of inputs and outputs')
            for fb_ii in feedback:
                check_binary(feedback[fb_ii])
            for jj, G_ii_jj in enumerate(G[ii]):
                if use_feedback[ii, jj] == 1 and ii not in feedback.keys():
                    raise ValueError(f'Use feedback is enabled for input {ii} output {jj}, but the feedback dict does not have a feedback polynomial')
                if use_feedback[ii, jj] == 0 and ii in feedback.keys():
                    if len(G_ii_jj) > 1:
                        if sum(G_ii_jj[1:]) > 0:
                            raise ValueError(f'Use feedback is disabled for input {ii} output {jj}, but the transfer function requires a FIR')
