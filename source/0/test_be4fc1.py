# https://github.com/ChistovPavel/QubitExperience/blob/4944c649be544b21140d31af1c85e46bec73d92b/QubitExperience/Test.py
import matplotlib.pyplot as plt
import BellsState as bs
import QuantumTeleportation as qt
import math
import Utils
import Grover

from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ, Aer, transpile, assemble

def plotHistogram(circuit):
    qasm_sim = Aer.get_backend('qasm_simulator')
    qobj = assemble(circuit)
    counts = qasm_sim.run(qobj).result().get_counts()
    plot_histogram(counts)

def bellsStateTest():
    qc = bs.getBellsState1([1,0], [1,0])
    stateVector = Utils.getStateVector(qc)
    Utils.printStateVector(stateVector, 4)
    qc.draw(output='mpl')
    plt.show()

def quantumTeleportationTest():
    qtc = qt.teleportateQuantumState([0.5, math.sqrt(1-0.25)])
    teleportationStateVector = Utils.getStateVector(qtc)
    qtc.draw(output='mpl')
    plotHistogram(qtc)
    plt.show()

def groverTest():
    grover_circuit = Grover.getGroverCircuit()
    grover_circuit.draw(output='mpl')
    plotHistogram(grover_circuit)
    plt.show()