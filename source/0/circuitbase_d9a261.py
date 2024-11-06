# https://github.com/stroblme/Visualization-SQripts/blob/47530a6f96e754531039ead1d476736cdc655642/circuitBase.py
from qiskit import QuantumCircuit, QuantumRegister, Aer, assemble, execute
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_state_city, array_to_latex
from numpy import pi
from colors import MAIN, WHITE

class circuitBase(QuantumCircuit):
    # def __init__(self, **kwargs):
        # super(QuantumCircuit, self).__init__(kwargs)

    def save(self, name, savePath='./out/'):
        self.draw(output='mpl', filename=f'{savePath}{name}.pdf', style={'displaycolor': {  'cx': (MAIN, WHITE),
                                                                                                                    'x': (MAIN, WHITE),
                                                                                                                    'measure': (MAIN, WHITE),
                                                                                                                    'swap': (MAIN, WHITE),
                                                                                                                    'h': (MAIN, WHITE)},
                                                                                                'subfontsize':14,
                                                                                                'fontsize':15})
