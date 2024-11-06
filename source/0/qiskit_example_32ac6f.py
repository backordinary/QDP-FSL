# https://github.com/HypsoHypso/QuantumComputingScripts/blob/53134783facba7d5972449b7ebb479d9439f3a98/qiskit_example.py
# Importing all the necessary library
from qiskit import QuantumCircuit, Aer, IBMQ, QuantumRegister, ClassicalRegister, execute
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import qiskit.tools.jupyter
import ipywidgets as widgets

# Layout
button_p = widgets.Button(
    description='Play')
gate_p = widgets.Dropdown(
    options=[('Identity', 'i'), ('Bit Flip', 'x')],
    description='Choice: ',
    disabled=False,
)
out_p = widgets.Output()
def on_button_clicked(b):
    with out_p:
        
        # Initial Circuit
        circuit_p = QuantumRegister(1, 'circuit')
        measure_p = ClassicalRegister(1, 'result')
        qc_p = QuantumCircuit(circuit_p, measure_p)
        
        # Turn 1
        qc_p.h(circuit_p[0])
        
        # Turn 2
        if gate_p.value == 'i':
            qc_p.i(circuit_p[0])
        if gate_p.value == 'x':
            qc_p.x(circuit_p[0])
        
        # Turn 3
        qc_p.h(circuit_p[0])
        
        # Measure  
        qc_p.measure(circuit_p, measure_p)
        
        # QASM
        backend_p = Aer.get_backend('aer_simulator')
        job_p = execute(qc_p, backend_p, shots=8192)
        res_p = job_p.result().get_counts()
        
        # Result
        if len(res_p) == 1 and list(res_p.keys())[0] == '0':
            print("You Lose to Quantum. Quantum Computer Wins")
        if len(res_p) == 1 and list(res_p.keys())[0] == '1':
            print("You Win against Quantum Computer")
        if len(res_p) == 2:
            print("Either Quantum or You Wins")

button_p.on_click(on_button_clicked)
widgets.VBox([gate_p, button_p, out_p])