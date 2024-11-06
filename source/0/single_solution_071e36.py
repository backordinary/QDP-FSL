# https://github.com/achieveordie/Grover-AI/blob/374a5579b478002a0114aa270a7bc61c4c83fdc8/qiskit-implementation/Oracles/single_solution.py
# To have all single-solution oracle circuits.
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit


class ThreeQubitsOracles(ABC):
    def __init__(self):
        self.n_qubits = 3
        self.oracle = QuantumCircuit(self.n_qubits)

    @abstractmethod
    def getOracle(self):
        pass


class Oracle0(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle0, self).__init__()
        self.search_int = 0

    def getOracle(self):
        for qubit in range(self.n_qubits):
            self.oracle.x(qubit)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(0, 1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        for qubit in range(self.n_qubits):
            self.oracle.x(qubit)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle1(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle1, self).__init__()
        self.search_int = 1

    def getOracle(self):
        for qubit in range(self.n_qubits-1):
            self.oracle.x(qubit)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(0, 1, self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        for qubit in range(self.n_qubits-1):
            self.oracle.x(qubit)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle2(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle2, self).__init__()
        self.search_int = 2

    def getOracle(self):
        self.oracle.x(0)
        self.oracle.x(self.n_qubits-1)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        self.oracle.x(0)
        self.oracle.x(self.n_qubits - 1)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle3(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle3, self).__init__()
        self.search_int = 3

    def getOracle(self):
        self.oracle.x(0)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=2)
        self.oracle.h(self.n_qubits-1)

        self.oracle.x(0)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle4(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle4, self).__init__()
        self.search_int = 4

    def getOracle(self):
        for i in range(1, self.n_qubits):
            self.oracle.x(i)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        for i in range(1, self.n_qubits):
            self.oracle.x(i)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle5(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle5, self).__init__()
        self.search_int = 5

    def getOracle(self):
        self.oracle.x(1)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        self.oracle.x(1)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle6(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle6, self).__init__()
        self.search_int = 6

    def getOracle(self):
        self.oracle.x(self.n_qubits-1)

        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        self.oracle.x(self.n_qubits-1)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate


class Oracle7(ThreeQubitsOracles):
    def __init__(self):
        super(Oracle7, self).__init__()
        self.search_int = 7

    def getOracle(self):
        self.oracle.h(self.n_qubits-1)
        self.oracle.ccx(control_qubit1=0, control_qubit2=1, target_qubit=self.n_qubits-1)
        self.oracle.h(self.n_qubits-1)

        oracle_gate = self.oracle.to_gate()
        oracle_gate.name = f"O({self.n_qubits})-{self.search_int}"
        return oracle_gate

# grover_circuit.measure_all()
#
# backend = Aer.get_backend('qasm_simulator')
# counts = execute(grover_circuit, backend=backend, shots=1024).result().get_counts()
# pprint(counts, indent=4)
