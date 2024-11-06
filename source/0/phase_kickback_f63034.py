# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/8a45f599c8c6d705d4435e253096ee18b1612c19/Qiskit/tutorials/phase_kickback.py

#%%
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_textbook.widgets import scalable_circuit
from numpy import pi
import warnings
warnings.filterwarnings("ignore")
from ..utils import plot_current_bloch_state
# %%
pk_a = QuantumCircuit(2)
pk_a.h(0)
pk_a.p(pi/4, 0)
pk_a.x(1)
plot_current_bloch_state(pk_a)
pk_a.cp(pi/4,0,1)
plot_current_bloch_state(pk_a)
pk_a.draw()
# %%