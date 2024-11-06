# https://github.com/jamesjcai/quantum/blob/26a1b389ce9002174d56cafe805d7b7245210f4e/%E9%87%8F%E5%AD%90/aaa.py
from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
# In Jupyter Notebooks we can display this nicely using Latex.
# If not using Jupyter Notebooks you may need to remove the 
# array_to_latex function and use print() instead.
# from qiskit_textbook.tools import array_to_latex

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0,1)
print(qc.draw())  # `display` is a command for Jupyter notebooks
                    # similar to `print`, but for rich content


qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
qc.h(1)
qc.draw()


# Let's see the result
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
print(final_state)

# ----------------

qc = QuantumCircuit(1)
qc.x(0)
qc.h(0)
qc.draw()

# Let's see the result
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()
print(final_state)