# https://github.com/micpap25/intro-quantum-computing-project/blob/733df3d3ea16b4fbed3492259e0a42bc3e66693b/string-to-circuit.py
# %%
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
# %%
# !!! Function taken from the
# Grover's Search Algorithm portion of the textbook
def display_unitary(qc, prefix=""):
    """Simulates a simple circuit and display its matrix representation.
    Args:
        qc (QuantumCircuit): The circuit to compile to a unitary matrix
        prefix (str): Optional LaTeX to be displayed before the matrix
    Returns:
        None (displays matrix as side effect)
    """
    from qiskit.visualization import array_to_latex
    sim = AerSimulator()
    # Next, we'll create a copy of the circuit and work on
    # that so we don't change anything as a side effect
    qc = qc.copy()
    # Tell the simulator to save the unitary matrix of this circuit
    qc.save_unitary()
    unitary = sim.run(qc).result().get_unitary()
    display(array_to_latex(unitary, prefix=prefix))
# %%
# !!! My code
# Information gathered from
# http://nbviewer.org/github/diemilio/grovers-algo/blob/master/Grovers_n-qubit_Oracle.ipynb
# Set up the string
STRING = "1111"
LENGTH = len(STRING)
# %%
# Create a multi-control z gate
multi_control_z = QuantumCircuit(name="mcz")
control_register = QuantumRegister(LENGTH - 1)
multi_control_z.add_register(control_register)
target_register = QuantumRegister(1)
multi_control_z.add_register(target_register)
multi_control_z.mcrz(np.pi,control_register,target_register)
# %%
# Create the oracle circuit
oracle = QuantumCircuit(LENGTH)
for i in range(LENGTH):
    if (STRING[-i - 1] == "0"):
        oracle.x(i)
oracle.append(multi_control_z, range(LENGTH))
for i in range(LENGTH):
    if (STRING[-i - 1] == "0"):
        oracle.x(i)
oracle.draw()
# %%
# !!!! All following code to run the oracle circuit
display_unitary(oracle, "U_\\text{oracle}=")
