# https://github.com/RobertJN64/Quantum/blob/169dbfdd05129c8e67d3869e6a406a39e7104ed5/Old%20Circuits/CircuitDemo.py
import QCircuitTools as qct
from matplotlib import pyplot
from qiskit.providers.aer import AerSimulator
from qiskit import visualization as vs
#import math

def run():
    circuitManager = qct.CircuitManager(2, measures=['PreMeasure'], qMeasures=["Qubit A", "Qubit B"])

    circuit = circuitManager.circuit

    circuit.h(0)
    circuitManager.measure(0, 'PreMeasure')
    circuit.cx(0,1)
    circuit.h(0)
    circuit.h(1)
    circuitManager.measureAll(barrier=True)

    circuitManager.simulate(AerSimulator(), 1000)

    # Draw the circuit
    fig = pyplot.figure()
    plta = fig.add_subplot(2,1,1)
    circuit.draw(output='mpl', ax=plta)
    pyplot.get_current_fig_manager().set_window_title('Circuit')
    circuitManager.printEntanglementTable()
    circuitManager.printEntanglements()
    pltb = fig.add_subplot(2,1,2)
    vs.plot_histogram(circuitManager.counts, ax=pltb)
    pyplot.show()

