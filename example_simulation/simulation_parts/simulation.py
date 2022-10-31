from example_simulation.simulation_parts.full_link import FullLink
from example_simulation.analyzer.analyzer_classes import Checker, Memory
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, link_data, verbose=False):
        self.Link  = FullLink(link_data, verbose)
        self.BitChecker = Checker(link_data.prbs_data)
        self.Memory     = Memory()

    def gather_memory(self):
        self.Memory.tx_prbs_chunks.append(self.Link.Tx.prbs_chunk.copy())
        self.Memory.tx_encoded_prbs_chunks.append(self.Link.Tx.encoded_prbs_chunk.copy())
        self.Memory.tx_mapped_prbs_chunks.append(self.Link.Tx.mapped_prbs_chunk.copy())
        self.Memory.ch_out_chunks.append(self.Link.Ch.ch_out.copy())
        self.Memory.rx_ctle_out_chunks.append(self.Link.Rx.ctle_out.copy())
        self.Memory.rx_adc_out_chunks.append(self.Link.Rx.adc_out.copy())
        self.Memory.rx_slicer_in_chunks.append(self.Link.Rx.slicer_in.copy())
        self.Memory.rx_slicer_out_chunks.append(self.Link.Rx.slicer_out.copy())
        self.Memory.rx_demapped_chunks.append(self.Link.Rx.demapped_chunk.copy())
        self.Memory.rx_decoded_chunks.append(self.Link.Rx.decoded_chunk.copy())

        self.Memory.cdr_phase_vec.append(self.Link.Rx.phase)
        self.Memory.lms_mse_vec.append(self.Link.Rx.lms_mse_last)
        self.Memory.lms_ffe_vecs.append(self.Link.Rx.ffe_dfe.ffe_vec.copy())
        self.Memory.lms_dfe_vecs.append(self.Link.Rx.ffe_dfe.dfe_vec.copy())

    def perform_convergence(self, verbose=False):
        self.Link()
        self.Memory.reset()
        self.Link.start_convergence()
        count = 0
        while not self.Link.Rx.converge_done:
            count += 1
            if verbose and count % 10 == 0:
                print(f'Chunk number {count:d}')
            self.Link()
            self.gather_memory()
            if count % 128 == 0:
                plt.figure()
                plt.plot(self.Memory.cdr_phase_vec)
                plt.figure()
                plt.plot(10 * np.log10(self.Memory.lms_mse_vec))
                plt.figure()
                plt.plot(np.concatenate(self.Memory.rx_slicer_in_chunks), '.')
                plt.show()
                print('hi')



