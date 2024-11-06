# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/test_python/host.py
#import qsharp
#from Qrng import SampleRngGenerator
from qiskit.visualization import plot_bloch_vector, plot_state_qsphere, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
import numpy as np

def show_figure(fig):
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show()

def DJ_StepByStep(n, oracle, title, multivector=True):
    init_state = ''
    for i in range(n+1):
        init_state += '0'
    
    # Create initial state
    u0 = Statevector.from_label(init_state)
    circuit_init = QuantumCircuit(n+1)
    circuit_init.x(n)

    # Create initial state (|00 ... 01>)
    u0 = u0.evolve(circuit_init)
    if multivector == True:
        fig_u0 = plot_bloch_multivector(u0, title="|u0>")
    else:
        fig_u0 = plot_state_qsphere(u0.data)

    # Apply hadamard gate to all qubits
    circuit_hadamard_in = QuantumCircuit(n+1)
    for i in range(n+1):
        circuit_hadamard_in.h(i)
    u1 = u0.evolve(circuit_hadamard_in)
    if multivector == True:
        fig_u1 = plot_bloch_multivector(u1, title="|u1> = H|u0>")
    else:
        fig_u1 = plot_state_qsphere(u1.data)

    # Apply the oracle
    u2 = u1.evolve(oracle)
    if multivector == True:
        fig_u2 = plot_bloch_multivector(u2, title="|u2> = Uf|u1>, " + title)
    else:
        fig_u2 = plot_state_qsphere(u2.data)

    # Apply hadamard
    circuit_hadamard_out = QuantumCircuit(n+1)
    for i in range(n):
        circuit_hadamard_out.h(i)
    u3 = u2.evolve(circuit_hadamard_out)
    if multivector == True:
        fig_u3 = plot_bloch_multivector(u3, title="|u3> = H|u2>")
    else:
        fig_u3 = plot_state_qsphere(u3.data)

    return fig_u0


#myOracle = QuantumCircuit(4)
# myOracle.z(0)
# myOracle.z(2)
# fig = DJ_StepByStep(4, myOracle, "Uf = z(0) et z(2)", False)
# show_figure(fig)

init_state = '000'
u0 = Statevector.from_label(init_state)
#fig = plot_bloch_multivector(u0, title="QFT transform")
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '001'
u0 = Statevector.from_label(init_state)
#fig = plot_bloch_multivector(u0, title="QFT transform")
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '010'
u0 = Statevector.from_label(init_state)
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '011'
u0 = Statevector.from_label(init_state)
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '100'
u0 = Statevector.from_label(init_state)
#fig = plot_bloch_multivector(u0, title="QFT transform")
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '101'
u0 = Statevector.from_label(init_state)
#fig = plot_bloch_multivector(u0, title="QFT transform")
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '110'
u0 = Statevector.from_label(init_state)
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

init_state = '111'
u0 = Statevector.from_label(init_state)
test_qft = QFT(3)
u1=u0.evolve(test_qft)
fig = plot_bloch_multivector(u1, title="QFT transform")

show_figure(fig)