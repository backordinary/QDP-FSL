# https://github.com/Appsolutely-Wonderful/Quantum/blob/68658a8eba1e6ba7b9cfaa3ec8ec8285423a1f19/qasm_simulator.py
from qiskit import Aer, QuantumCircuit, assemble
from qiskit.result import Result
from qiskit.qobj import QasmQobj

class QasmSimuluator:
    """
    Wrapper for the "qasm_simulator" back end
    """

    def __init__(self):
        """
        Initializes the QasmSimulator
        """
        self.sim = Aer.get_backend('qasm_simulator')
    
    def run_circuit(self, circuit: QuantumCircuit) -> Result:
        """
        Runs a given QuantumCircuit.
        This assembles your quantum circut and executes
        self.run()
        """
        qobj = assemble(circuit)
        return self.run(qobj)

    def run(self, qobj: QasmQobj) -> Result:
        """
        Executes a QasmObject in the Qasm Simulator
        """
        result = self.sim.run(qobj).result()
        return result
