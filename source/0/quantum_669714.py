# https://github.com/vipulsnk/python_boiler_plate/blob/590a1eb4b99eb4cd6cd40d1750b1cd2263c5ad2b/core/quantum.py
import matplotlib as mpl
from qiskit import QuantumCircuit, circuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt



# # plt.show()


def create_state():
  circuit = QuantumCircuit(1, 2)
  circuit.h([0])
  return circuit

def draw_circuit(circuit):
  fig = circuit.draw(output="mpl")
  plt.savefig("./snaps/circuit.jpeg")
  
