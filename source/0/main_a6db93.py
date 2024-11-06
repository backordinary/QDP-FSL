# https://github.com/Baccios/CTC_iterator/blob/02c77c2a8e882e48df6ef1dcd186f9124ecaad38/main.py
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

from qiskit import IBMQ
# from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy

from ctc.brun import get_u
from ctc.simulation import CTCCircuitSimulator
from math import sqrt
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # provider = IBMQ.load_account()
    # backend = provider.get_backend('ibmq_santiago')

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    # backend = provider.get_backend('ibmq_lima')
    
    backend = least_busy(provider.backends(
                                        filters=lambda x: x.configuration().n_qubits >= 4
                                        and not x.configuration().simulator
                                        and x.status().operational is True
                                        and 'reset' in x.configuration().supported_instructions))
                                        
    print("least busy backend: ", backend)
    # backend = QasmSimulator.from_backend(backend)
    

    
    c_values_2bit = ["01", 0.5]

    c_tick_labels_2bits = ["|c⟩=|01⟩", "|c⟩=|++⟩"]

    c_tics_cbs_vs_h = ["|0001⟩", "|++++⟩", "|0+++⟩", "|00++⟩", "|000+⟩", "|1000⟩"]

    sim = CTCCircuitSimulator(size=2, k_value=3)
    start = time.time()
    sim.test_c_variability(c_values_2bit, 1, 21, 2, c_tick_labels=c_tick_labels_2bits, plot_d=2)
    end = time.time()
    print("elapsed time = ", end-start, "s")
    
    

    sim = CTCCircuitSimulator(size=2, k_value=2, ctc_recipe="brun")
    start = time.time()
    sim.test_convergence(c_value=0.5, start=1, stop=17, step=2, cloning="no_cloning", backend=backend)
    end = time.time()
    print("elapsed time = ", end-start, "s")

    
    

    c_values = ["0000", "0110", "1001", "1111"]
    k_values = [1, 7, 8, 14]
    times = []

    for c in c_values:
        for k in k_values:
            sim = CTCCircuitSimulator(size=4, k_value=k, cloning_size=3)
            start = time.time()
            sim.test_convergence(c_value=c, start=1, stop=361, step=30, cloning="no_cloning")
            end = time.time()
            times.append(end-start)
            print("elapsed time = ", end - start, "s")

    print("Execution times = ", times)
    
    """
    start = time.time()

    for k in range(16):
        sim = CTCCircuitSimulator(size=4, k_value=k, cloning_size=3, ctc_recipe="nbp")
        for c in range(16):
            if c != k:
                sim.test_convergence(c_value=c, start=1, stop=91, step=10, cloning="no_cloning")

    for k in range(16):
        sim = CTCCircuitSimulator(size=4, k_value=k, cloning_size=3, ctc_recipe="nbp")
        sim.test_convergence(c_value=0.5, start=1, stop=91, step=10, cloning="no_cloning")

    end = time.time()
    print("elapsed time = ", end - start, "s")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
