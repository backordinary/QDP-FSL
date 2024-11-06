# https://github.com/avirajs/QuantumROM/blob/a3841331b7d3047a1a4b15b2cf5c7f7249d2cb1c/angle_QROM_code/angle_QROM_code/angle_QROM_test.py
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from angle_QROM_baseline import get_angle_QROM
from angle_QROM_optimized import get_angle_optim_QROM
import numpy as np

def plot_results(qc, plot_ones = False):

    import qiskit
    from qiskit.providers.aer import QasmSimulator
    from qiskit import QuantumCircuit, assemble, Aer
    from math import pi, sqrt
    from qiskit.visualization import plot_bloch_multivector, plot_histogram

    shot_num = 100000
    sim = QasmSimulator(method='statevector')
    # # Perform an ideal simulation
    result_ideal = qiskit.execute(qc, sim, shots=shot_num).result()
    counts_ideal = result_ideal.get_counts()
    # print('Counts(ideal):', counts_ideal)

    counts_ideal_format = {}
    for key in counts_ideal.keys():
        new_key = key[::-1]
        if plot_ones and key[-1]=="0":
            continue
        counts_ideal_format[new_key] = counts_ideal[key]

    print('Counts(ideal):', counts_ideal_format)

    return counts_ideal_format


def get_verification_results(qc_og, qc_min):
    import qiskit
    from qiskit.providers.aer import QasmSimulator
    from qiskit import QuantumCircuit, assemble, Aer
    from math import pi, sqrt
    from qiskit.visualization import plot_bloch_multivector, plot_histogram
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister






    control_size = len(qc_og.qubits)-1
    addr_bits = QuantumRegister(control_size,"addr")
    data_bits = QuantumRegister(1,"data")
    ca = ClassicalRegister(control_size,"ca_m")
    cm = ClassicalRegister(1,"data_m")
    qc = QuantumCircuit(addr_bits,data_bits,cm,ca)

    qc.compose(qc_og, inplace=True)

    qc.compose(qc_min.inverse(), inplace=True)



    qc.measure(addr_bits,ca)

    qc.measure(data_bits,cm)


    sim = QasmSimulator(method='statevector')
    # # Perform an ideal simulation
    result_ideal = qiskit.execute(qc, sim, shots=10000).result()
    counts_ideal = result_ideal.get_counts()

    return counts_ideal
