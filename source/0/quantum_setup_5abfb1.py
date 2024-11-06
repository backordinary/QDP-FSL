# https://github.com/kleinfossil/quantom_test/blob/2357a8f19f3b631670f807c095b4697dce7799b3/quantum_setup.py
from qiskit import *
from qiskit.aqua import QuantumInstance
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
import numpy as np
# import pylatexenc
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy


# Prints out qiskit version and IBM Account Infos
def qiskitversion():
    print('Current qiskit Version: ', qiskit.__qiskit_version__)
    print('Current IBM Account: ', IBMQ.load_account())


# Executes a specific quantum Circut
def execute_quantum_circuit(cir, shots=100, number_qubits=None, use_ibmq=False, specific_ibmq_backend=None):

    # Select a specific quantum circuit
    quantum_instance = get_quantumcomputer_quantum_instance(shots, number_qubits, use_ibmq, specific_ibmq_backend)

    # Execute quantum circut
    result = quantum_instance.execute(cir)
    plot_histogram(result.get_counts(cir))

    return quantum_instance.backend_name


# Selects a quantom computer instance. This can be ibmq or a local simulator
def get_quantumcomputer_quantum_instance(shots=100, number_qubits=None, use_ibmq=False, specific_ibmq_backend=None):
    if use_ibmq:
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        if specific_ibmq_backend is None:
            if number_qubits is not None:
                if 0 < number_qubits < 5:
                    # Selects the lest busy backend which has enough number_of_qubits and is not a simulator and is operational
                    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (number_qubits + 1) and not x.configuration().simulator and x.status().operational == True))
                    print("The least busy backend ", str(backend))
                    print("The following Backend has be selected: ", backend)
                    quantum_instance = QuantumInstance(backend, shots, skip_qobj_validation=False)
                    return quantum_instance
                else:
                    if 5 <= number_qubits < 32:
                        # selects the IBMQ simulator
                        backend = provider.get_backend('ibmq_qasm_simulator')
                        print("The following Backend has be selected: ", str(backend))
                        quantum_instance = QuantumInstance(backend, shots, skip_qobj_validation=False)
                        return quantum_instance
                    else:
                        raise Exception(
                            'Sorry your number of qubits reached. The max number for IBMQ is 31. Your requested: ' + str(number_qubits))
            else:
                # selects the IBMQ simulator
                backend = provider.get_backend('ibmq_qasm_simulator')
                print("The following Backend has be selected: ", str(backend))
                quantum_instance = QuantumInstance(backend, shots, skip_qobj_validation=False)
                return quantum_instance
        else:
            # Selects a specific backend on IBMQ Backend
            try:
                backend = provider.get_backend(specific_ibmq_backend)
            except Exception as e:
                print("Stacktrace: " + str(e))
                print("The backend could not be selected. Please check again the backend name you have given: " + str(specific_ibmq_backend))
            print("The following Backend has be selected: ", str(backend))
            quantum_instance = QuantumInstance(backend, shots, skip_qobj_validation=False)
            return quantum_instance

    else:
        # Choose a local Simulator
        simulator = Aer.get_backend('qasm_simulator')
        simulator_instance = QuantumInstance(simulator, shots)
        print("The following simulator was selected: " + str(simulator_instance.backend_name))
        return simulator_instance


def get_simple_example_quantum_circuit():
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    simple_circuit = QuantumCircuit(qr, cr)
    simple_circuit.h(qr[0])
    simple_circuit.cx(qr[0], qr[1])
    simple_circuit.measure(qr, cr)

    simple_circuit.draw(output='mpl')
    return simple_circuit


if __name__ == '__main__':
    # --Print qiskit version--
    qiskitversion()

    # --Prepatation Step--
    circuit_trys = 1024
    # number_of_qubits = 3

    # --create a simple quantum circuit--
    cir = get_simple_example_quantum_circuit()

    # --Execute on the quantom simulator--
    execute_quantum_circuit(cir=cir, shots=circuit_trys, use_ibmq=True)

    plt.show()

