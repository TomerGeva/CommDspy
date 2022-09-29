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
        # ==============================================================================================================
        # Extracting the constraint length and memory array
        # ==============================================================================================================
        memory = []  # hold the number of memory registers in the coding scheme
        constraint_len = 0
        for ii in G:
            memory.append(G[ii].shape[1] - 1)
            if G[ii].shape[1] > constraint_len:
                constraint_len = G[ii].shape[1]
        memory_cumsum = np.concatenate([[0], np.cumsum(memory)])
        # ==============================================================================================================
        # Extracting the constraint length and memory array
        # ==============================================================================================================
        self.num_states = 2 ** sum(memory)
        self.inputs = get_bin_perm(self.n_in)
        self.states = get_bin_perm(sum(memory))
        # ==============================================================================================================
        # Creating the trellis
        # ==============================================================================================================
        self.trellis, self.io_dict = create_trellis(G, self.states, self.inputs, memory_cumsum) #, feedback=feedback, use_feedback=use_feedback)


def create_trellis(G, states, inputs, memory_cumsum, feedback=None, use_feedback=None):
    # ==================================================================================================================
    # Local variables
    # ==================================================================================================================
    n_in  = len(G)
    n_out = G[0].shape[0]
    trellis_dict = {}
    io_dict      = {}  # keys are (out_state, in_state), values are (in, out)
    if feedback is None:
        no_feedback  = True
        feedback     = {}
        use_feedback = np.zeros([n_in, n_out], dtype=int)
    else:
        no_feedback  = False
    # ==================================================================================================================
    # Building the trellis
    # ==================================================================================================================
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
        for input_vec in inputs:
            key = (tuple(input_vec), tuple(state))
            # **************************************************************************************************
            # For each state and input, we compute the outputs and the next state ignoring feedback at the moment
            # **************************************************************************************************
            c_vec      = np.zeros(n_out, dtype=int)
            next_state = np.zeros_like(state)
            for kk, in_k in enumerate(input_vec):
                G_kk     = G[kk]
                # _________ performing the feedback if needed ______________
                if kk in feedback.keys():
                    in_k_fb = (in_k + memory_dict[kk][:-1].dot(feedback[kk][1:])) % 2
                else:
                    in_k_fb = in_k
                memory_k = np.concatenate([[in_k_fb], memory_dict[kk]])
                # memory_k = np.concatenate([[in_k], memory_dict[kk]])
                # _____________   next state computation  __________________
                next_state_k = memory_k[:-1]
                next_state[memory_cumsum[kk]:memory_cumsum[kk + 1]] = next_state_k
                # _____________ output vector computation __________________
                if no_feedback:
                    c_vec = c_vec ^ (G_kk.dot(memory_k) % 2)
                else:
                    for jj in range(n_out):
                        G_kk_jj = G[kk][jj]
                        if len(G_kk_jj) == 1 or use_feedback[kk, jj] == 0:
                            # no memory, either systematic or always 0 output, depending on G_ii_jj[0] value.
                            # 1 - systematic ; 0 - zero output
                            c_vec[jj] = c_vec[jj] ^ (in_k * G_kk_jj[0])
                        else:
                            c_vec[jj] = c_vec[jj] ^ (G_kk_jj.dot(memory_k) % 2)
            # **************************************************************************************************
            # Adding to trellis dict
            # **************************************************************************************************
            trellis_dict[key] = (tuple(c_vec), tuple(next_state))
            io_key            = (tuple(tuple(state)), tuple(next_state))
            io_dict[io_key]   = (tuple(input_vec), tuple(c_vec))

    return trellis_dict, io_dict


if __name__ == '__main__':
    G = {0: np.array([[1, 0, 1], [1, 1, 1]])}
    trellis, n_states = create_trellis(G)
    print('hi')