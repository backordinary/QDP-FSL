# https://github.com/vlcanunal/Q-Bonibon/blob/6e7ae26f05ed53a652999247bc1633c5c513538f/app.py
from flask import Flask
from flask import send_file
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit #, Aer, execute
from numpy import pi
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from qiskit.visualization.state_visualization import plot_bloch_multivector

QUANTUM_BITS = 2
CLASSICAL_BITS = 2

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello world'

class Circuit:
    def __init__(self, qb, cb):
        self.count = 0
        self.qr = QuantumRegister(qb)
        self.cr = ClassicalRegister(cb)
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self.circuitHistory = []

    def applyGate(self, gateName, qubitId):
        # self.circuitHistory.append(self.circuit)
        if(gateName == 'h'):
            self.circuit.h(self.qr[qubitId])
        elif(gateName == 'x'):
            self.circuit.x(self.qr[qubitId])
        self.count += 1

    def getCircuit(self):
        fileName = f'circuit-{self.count}.png'
        self.circuit.draw(output='mpl',filename=fileName)
        # simulator = Aer.get_backend('unitary_simulator')
        # result = execute(circuit, backend = simulator).result()
        # statevector = result.get_unitary()
        # plt.plot(plot_bloch_multivector(statevector))
        return send_file(fileName)

circuit = Circuit(QUANTUM_BITS, CLASSICAL_BITS)


    
gateName = 'h'
@app.route('/gate')
def gateSelect(gateName = gateName):
    circuit.applyGate(gateName, 0)
    return circuit.getCircuit()

if __name__ == '__main__':
    app.run()