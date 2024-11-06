# https://github.com/lihaifeng0710/PythonGUI/blob/10a1fce409bb3cccb1b9ba51d8d3177b6df7e9fb/%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/%E7%BB%98%E5%9B%BE.py
from qiskit import *
# For Jupyter Notebooks:%config InlineBackend.figure_format = 'svg' # Makes the images look nice
qc = QuantumCircuit()
qr = QuantumRegister(2,'qreg')
qc.add_register( qr )
qc.h(qr[0])
qc.cx(qr[0], qr[1])


