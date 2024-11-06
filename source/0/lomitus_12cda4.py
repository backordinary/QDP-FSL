# https://github.com/markusharjoittelu/lomittaja/blob/0659301d8800a5b0b572e0e11f1c442a3bf3a49a/lomitus.py
from flask import Flask
import numpy as np
from qiskit import(QuantumCircuit, execute, Aer)
from qiskit.visualization import plot_histogram

main_page = """
<!DOCTYPE html>
<html>
<body>

<h2>Lomittaja</h2>
<form action="">
    <button type="submit">Lomita lisää!</button>
<form>
 
</body>
</html>
"""

app = Flask(__name__)

@app.route('/')
def index():
    qc = QuantumCircuit(2)
    # Apply H-gate to the first:
    qc.h(0)
    # Apply a CNOT:
    qc.cx(0,1)
    return main_page

if __name__ == '__main__':
    app.run(debug=True, port=5000)