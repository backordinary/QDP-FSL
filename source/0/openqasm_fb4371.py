# https://github.com/samu-magaz/quantum-factory-method/blob/f0ed6aaf9332c4ecc61e508a910d394a175856ca/examples/openqasm.py
'''OpenQASM example module
'''

import abc

import qiskit
from qiskit_aer import Aer

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

class Circuit(abc.ABC):
    '''Base class for Circuit classes
    '''
    pass


class QiskitCircuit(Circuit):
    '''Circuit class for Qiskit
    '''
    def __init__(self, built_circuit: qiskit.QuantumCircuit) -> None:
        self.built_circuit = built_circuit


class CirqCircuit(Circuit):
    '''Circuit class for Cirq
    '''
    def __init__(self, built_circuit: cirq.Circuit) -> None:
        self.built_circuit = built_circuit


class Result:
    '''Class to store the values after executing a circuit
    '''
    def __init__(self, values: list[float]) -> None:
        self.values = values


class Platform(abc.ABC):
    '''Base class for Platform classes
    '''
    @staticmethod
    @abc.abstractmethod
    def build(openqasm_str: str) -> Circuit:
        '''Abstract method for building the corresponding quantum circuit of an OpenQASM string

        :param openqasm_str: The OpenQASM string defining a quantum circuit
        :type openqasm_str: str
        :return: The Circuit object with the quantum circuit
        :rtype: Circuit
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def execute(circuit: Circuit) -> Result:
        '''Abstract method for executing a previously built quantum circuit

        :param circuit: The Circuit object containing the quantum circuit that will be executed
        :type circuit: Circuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        pass


class QiskitPlatform(Platform):
    '''Platform class for Qiskit
    '''
    shots = 1024

    @staticmethod
    def build(openqasm_str: str) -> QiskitCircuit:
        '''Builds the corresponding quantum circuit of an OpenQASM string

        :param openqasm_str: The OpenQASM string defining a quantum circuit
        :type openqasm_str: str
        :return: The QiskitCircuit object with the quantum circuit
        :rtype: QiskitCircuit
        '''
        return QiskitCircuit(qiskit.QuantumCircuit.from_qasm_str(openqasm_str))

    @staticmethod
    def execute(quantum_circuit: QiskitCircuit) -> Result:
        '''Executes a previously built quantum circuit

        :param quantum_circuit: The QiskitCircuit object containing the quantum circuit that will be executed
        :type quantum_circuit: QiskitCircuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        circ = quantum_circuit.built_circuit

        simulator = Aer.get_backend("aer_simulator")
        circ = qiskit.transpile(circ, simulator)
        result = simulator.run(circ, shots=QiskitPlatform.shots).result()

        counts = result.get_counts()
        counts = {int(k[0]): v for k,v in counts.items()}

        values = list(range(len(counts)))
        for k,v in counts.items():
            values[k] = v / QiskitPlatform.shots
        return Result(values)


class CirqPlatform(Platform):
    '''Platform class for Cirq
    '''
    shots = 1024

    @staticmethod
    def build(openqasm_str: str) -> CirqCircuit:
        '''Builds the corresponding quantum circuit of an OpenQASM string

        :param openqasm_str: The OpenQASM string defining a quantum circuit
        :type openqasm_str: str
        :return: The CirqCircuit object with the quantum circuit
        :rtype: CirqCircuit
        '''
        return CirqCircuit(circuit_from_qasm(openqasm_str))

    @staticmethod
    def execute(quantum_circuit: CirqCircuit) -> Result:
        '''Executes a previously built quantum circuit

        :param quantum_circuit: The CirqCircuit object containing the quantum circuit that will be executed
        :type quantum_circuit: CirqCircuit
        :return: The Result object with the values of the execution
        :rtype: Result
        '''
        circ = quantum_circuit.built_circuit

        result = cirq.Simulator().run(circ, repetitions=CirqPlatform.shots)

        counts = result.measurements

        values = list(range(len(counts)))
        temp = 0
        for [v] in list(counts.items())[0][1]:
            temp += v
        values[0] = temp / CirqPlatform.shots
        values[1] = (CirqPlatform.shots - temp) / CirqPlatform.shots
        return Result(values)
