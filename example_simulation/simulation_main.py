from example_simulation.simulation_parts.simulation import Simulation
from config_file import *

def main():
    np.random.seed(140993)
    Sim  = Simulation(Link_data, verbose=Control_vars.simulation_verbose)
    # Test to see that the generator is OK
    # prbs = []
    # for ii in range(128):
    #     _ = Tx.generate()
    #     prbs = prbs + list(Tx.prbs_chunk)
    # print(np.allclose(np.array(prbs)[:-1], BitChecker.prbs_vec))
    Sim.perform_convergence(verbose=True)
    # for ii in range(30):
    #     print(f'Chunk number {ii:d}')
    #     received_data = Full_link()
    #     if ii == 1:
    #         Full_link.start_convergence()
    #     if Full_link.Rx.converge_done:
    #         chunk_ber = BitChecker.check_ber(received_data)
    #         print(f'Chunk BER is {chunk_ber:.2e} \nAccumulated BER is {BitChecker.acc_errors/BitChecker.acc_symbols:.2e}')
    print('hi')


if __name__ == '__main__':
    main()