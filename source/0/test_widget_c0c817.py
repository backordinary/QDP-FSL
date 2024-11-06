# https://github.com/dowusu-antwi/qaoa-research/blob/641b4a835e992397545fa1438363929d1cba9e66/update/test_widget.py
#!/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit.visualization.matplotlib import MatplotlibDrawer as Drawer
from qiskit.visualization.utils import _get_layered_instructions as UtilOutput
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import sys
from random import random

class Window(QtWidgets.QWidget):
    def __init__(self):
        self.application_manager = QtWidgets.QApplication(sys.argv)
        super().__init__()
        self.resize(1000, 1000)
        self.layout = self.build()
        self.timer = self.setup_timer()
        self.visual_circuit = None
        self.visual_circuit_canvas = None


    def resize(self, width, height):
        """
        Change size of console window.
        """
        self.setGeometry(0, 0, width, height)


    def build(self):
        """
        Builds window.
        """
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)  
        return layout


    def setup_timer(self):
        """
        Builds timer.
        """
        timer = QtCore.QTimer(self)
        TIMESTEP = 2000
        timer.start(TIMESTEP)
        timer.timeout.connect(self.update_circuit)
        return timer


    def embed_circuit(self, visual_circuit):
        """
        Embeds circuit image into window.
        """
        layout = self.layout
        visual_circuit_canvas = Canvas(visual_circuit.image)
        layout.addWidget(visual_circuit_canvas, *(0, 0, 1, 1))
        visual_circuit.image.tight_layout()
        self.visual_circuit = visual_circuit
        self.visual_circuit_canvas = visual_circuit_canvas


    def update_circuit(self):
        """
        Updates visual circuit image.
        """
        visual_circuit = self.visual_circuit
        visual_circuit_canvas = self.visual_circuit_canvas
        circuit_image = visual_circuit.image
        circuit_axes = visual_circuit.axes
        position = circuit_axes.get_position()
        circuit_axes.clear()
        visual_circuit_canvas.draw()
        circuit_image.tight_layout()
        circuit = visual_circuit.build()
        qubits, clbits, nodes = UtilOutput(circuit,
                                           reverse_bits=False,
                                           justify=None,
                                           idle_wires=True)
        drawing = Drawer(qubits,
                         clbits,
                         nodes,
                         ax=circuit_axes,
                         qregs=circuit.qregs,
                         cregs=circuit.cregs)
        drawing.draw()
        circuit_axes.set_position([0.0,
                                   0.0,
                                   position.width,
                                   position.height])
        visual_circuit_canvas.draw()
 

    def run(self):
        """
        Opens window.
        """
        self.show()
        self.application_manager.exec_()

 
class VisualCircuit():
    def __init__(self, max_circuit_size):
        self.max_circuit_size = max_circuit_size
        self.gates = [lambda circuit, q, _: circuit.h(q),
                      lambda circuit, q1, q2: circuit.cx(q1, q2),
                      lambda circuit, q1, q2: circuit.cz(q1, q2)]
        circuit = self.build(max_circuit_size)
        self.image, self.axes = self.get_figure(circuit)


    def build(self, max_circuit_size=None):
        """
        Builds a randomly-sized circuit, given maximum size (used to bound both
         number of qubits and number of gates).
        """
        if max_circuit_size == None:
            max_circuit_size = self.max_circuit_size
        circuit_size = 2 + int(random() * (max_circuit_size))
        circuit = QuantumCircuit(circuit_size, circuit_size)
        gates = self.gates
        num_gates = len(gates)
        for count in range(max_circuit_size):
            q1, q2 = self.get_qubits(circuit_size)
            gates[int(random() * num_gates)](circuit, q1, q2)
        circuit.measure_all()
        return circuit


    def get_figure(self, circuit):
        """
        Get circuit figure objects.
        """ 
        image = circuit.draw(output="mpl")
        axes = image.get_axes()[0]
        return image, axes


    def get_qubits(self, circuit_size):
        """
        Gets two nonequal valid qubits.
        """
        q1 = int(random() * circuit_size)
        q2 = int(random() * circuit_size)
        while q1 == q2:
            q2 = int(random() * circuit_size)
        return q1, q2
        

def main():
    """
    Makes visual circuit and displays, updating circuit every timestep.
    """
    window = Window()
    visual_circuit = VisualCircuit(10)
    window.embed_circuit(visual_circuit)
    window.run()


if __name__ == "__main__":
    main()
