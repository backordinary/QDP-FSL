# https://github.com/seendsouza/patrickbot/blob/11898472b27e08a0933f3dbc2aa37d584d108717/src/observequantum.py
"""
Compares a user's guess to the state of a qubit after observation
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import available_backends, execute

def observe_equal_superposition():
    """
    This function observes a qubit in equal quantum superposition.
    Don't question the practicality of this.
    @return: observed state (0 or 1)
    """
    # TODO: This is deprecated, so consider updating it.
    # Create a Quantum Register with 1 qubit.
    q = QuantumRegister(1)
    # Create a Classical Register with 1 bit.
    c = ClassicalRegister(1)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, c)
    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc.h(q[0])
    # Add a Measure gate to see the state.
    qc.measure(q, c)
    # Compile and run the Quantum circuit on a simulator backend
    job_sim = execute(qc, "local_qasm_simulator",shots = 1)
    sim_result = job_sim.result()
    return int(next(iter(sim_result.get_counts(qc))))

class ObserveQuantum():
    """
    This class takes in a user's guess and compares it with the state of a qubit after observation.
    The guess is mapped to either 0 or 1. If the guess' mapped number matches the observed state,
    the observed() function returns True, otherwise False.
    """
    def __init__(self,observe_arguments):
        self.observe_arguments = observe_arguments

    def observed(self,guess):
        if self.observe_arguments[0] in guess:
            if observe_equal_superposition() == 0:
                return True
            else:
                return False
        elif self.observe_arguments[1] in guess:
            if observe_equal_superposition() == 1:
                return True
            else:
                return False
        else:
            raise Exception
