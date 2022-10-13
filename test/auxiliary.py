import csv
import numpy as np

def read_1line_csv(filename, delimiter=','):
    """
    :param filename: Input filename
    :param delimiter: The delimiter in the file
    :return: the first line of the file
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        ref_prbs_bin = np.array(next(reader)).astype(int)
    return ref_prbs_bin