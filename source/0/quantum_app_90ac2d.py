# https://github.com/PedruuH/Computacao-Quantica/blob/5fa109c8fa7fb7e4ffdb0c7c7192a803c918244e/proj_final/quantum_app.py
from cv2 import Algorithm
import matplotlib.pyplot as plt
import qiskit as qk
import qiskit.aqua.algorithms as qkal
import qiskit.aqua.components.oracles as LEO
from qiskit.tools.visualization import plot_histogram
from apitoken import apitoken 

expression = '((Pedro & Fabio) | (Carlos & Alex)) & ~(Fabio & Alex)'
algorith = qkal.Grover(LEO.LogicalExpressionOracle(expression))
                  
simulator = qk.Aer.get_backend('qasm_simulator')
result = algorith.run(simulator)
plot_histogram(result['measurement'])
plt.show()