# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/hello_draw.py
import qiskit
from PIL import Image
from qiskit.tools.visualization import matplotlib_circuit_drawer

qr = qiskit.QuantumRegister(1)
cr = qiskit.ClassicalRegister(1)
program = qiskit.QuantumCircuit(qr, cr)
program.measure(qr, cr)

image = matplotlib_circuit_drawer(program)

Image._show(image)