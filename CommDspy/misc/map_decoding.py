import numpy as np

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

