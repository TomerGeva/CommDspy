import numpy as np
from CommDspy.auxiliary import get_bin_perm

def map_decoding(permutations, codebook, pattern_block, error_prob):
    p_err   = None
    hamming = np.sum(np.abs(codebook[:, None, :] - pattern_block[None, :, :]), axis=2)
    if error_prob:
        # ----------------------------------------------------------------------------------------------------------
        # Creating the error_prob vector
        # ----------------------------------------------------------------------------------------------------------
        p_err = np.zeros(pattern_block.shape[0])
        # ----------------------------------------------------------------------------------------------------------
        # Error correction
        # ----------------------------------------------------------------------------------------------------------
        min_hamming = np.min(hamming, axis=0)
        not_codeword = min_hamming > 0
        correct_idx = np.argmin(hamming[:, not_codeword], axis=0)
        pattern_block[not_codeword] = codebook[correct_idx]
        # ----------------------------------------------------------------------------------------------------------
        # Fixing the hamming matrix, filling p_err
        # ----------------------------------------------------------------------------------------------------------
        p_err[not_codeword] = 1 / np.sum(hamming[:, not_codeword] == min_hamming[not_codeword], axis=0)
        hamming[correct_idx, not_codeword] = 0
    decoded_idx = np.argmin(hamming, axis=0)
    decoded_blocks = permutations[decoded_idx]
    if error_prob:
        return np.reshape(decoded_blocks, -1), p_err
    else:
        return np.reshape(decoded_blocks, -1)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Trellis object and function
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Trellis:
    """
    :param G:
    :param feedback: ignored atm
    :param use_feedback: ignored atm
    :return: Creating a trellis dictionart where:
                - keys are tuples of (input, state)
                - values are tuples of (output, next_state)
    """
    def __init__(self, G, feedback=None, use_feedback=None):
        self.n_in = len(G)
        self.n_out = G[0].shape[0]
        # ==================================================================================================================
        # Extracting the constraint length and memory array
        # ==================================================================================================================
        memory = []  # hold the number of memory registers in the coding scheme
        constraint_len = 0
        for ii in G:
            memory.append(G[ii].shape[1] - 1)
            if G[ii].shape[1] > constraint_len:
                constraint_len = G[ii].shape[1]
        memory_cumsum = np.concatenate([[0], np.cumsum(memory)])
        # ==================================================================================================================
        # Extracting the constraint length and memory array
        # ==================================================================================================================
        self.num_states = 2 ** sum(memory)
        self.inputs = get_bin_perm(self.n_in)
        self.states = get_bin_perm(sum(memory))
        # ==================================================================================================================
        # Creating the trellis
        # ==================================================================================================================
        self.trellis = create_trellis(G, self.states, self.inputs, memory_cumsum)

def create_trellis(G, states, inputs, memory_cumsum):
    n_in  = len(G)
    n_out = G[0].shape[0]
    trellis_dict = {}
    for state in states:
        # ----------------------------------------------------------------------------------------------------------
        # Filling memory for this state
        # ----------------------------------------------------------------------------------------------------------
        memory_dict = {}
        for kk in range(n_in):
            memory_dict[kk] = state[memory_cumsum[kk]:memory_cumsum[kk+1]]
        # ----------------------------------------------------------------------------------------------------------
        # finding next_states and outputs for each input from the current state
        # ----------------------------------------------------------------------------------------------------------
        for input in inputs:
            key = (tuple(input), tuple(state))
            # **************************************************************************************************
            # For each state and input, we compute the outputs and the next state ignoring feedback at the moment
            # **************************************************************************************************
            c_vec      = np.zeros(n_out)
            next_state = np.zeros_like(state)
            for kk, in_k in enumerate(input):
                memory_k = np.concatenate([[in_k], memory_dict[kk]])
                G_kk     = G[kk]
                # _____________ output vector computation __________________
                c_vec    = (c_vec + G_kk.dot(memory_k)) % 2
                # _____________   next state computation  __________________
                next_state_k                                        = memory_k[:-1]
                next_state[memory_cumsum[kk]:memory_cumsum[kk + 1]] = next_state_k
            # **************************************************************************************************
            # Adding to trellis dict
            # **************************************************************************************************
            trellis_dict[key] = (tuple(c_vec.astype(int)), tuple(next_state))

    return trellis_dict


if __name__ == '__main__':
    G = {0: np.array([[1, 0, 1], [1, 1, 1]])}
    trellis, n_states = create_trellis(G)
    print('hi')