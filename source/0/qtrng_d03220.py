# https://github.com/lukaszbladziak5/Podpis-Cyfrowy-SHA/blob/1714b0b89f5f6f7067d496a4d3783e79605eefb3/qtrng.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute,IBMQ
import PySimpleGUI as gui

def generate_number():

    IBMQ.enable_account('083bf4a193c6292790f166e25ee0ec1b8683e1df3ece4a6f119ed7b77470b3ecbb9cedbc657fc136a25d4d565d7813f292b8f34af6212dde2a2c982e27136ec8')
    provider = IBMQ.get_provider(hub='ibm-q')

    backend = provider.get_backend('ibmq_qasm_simulator')



    shots = 20000

    q = QuantumRegister(9,'q')
    c = ClassicalRegister(9,'c')

    circuit = QuantumCircuit(q,c)

    circuit.x(q[0])
    circuit.ch(q[0], q[1:9])
    circuit.measure(q,c)

    #print(circuit)

    job = execute(circuit, backend, shots=20000)

    #job_monitor(job)
    counts = job.result().get_counts()
    countsNew = {}
    for i in counts:
        b = int(i, 2)
        b >>= 1
        countsNew[b] = counts[i]

    counts = countsNew
    counts = sorted(counts.items())
    counts.reverse()
    a, b = counts[0]
    print(b)
    return b

def create_generatorGUI(number):
    
    gui.theme('BlueMono')
    # layout = [
    #     [gui.Text('This is your generated value: ' + str(number))],
    #     [gui.Submit('Ok')]
    # ]
    # window = gui.Window('RSA Generator', layout)
    # while True:
    #     event = window.read()
    #     if event == gui.WIN_CLOSED or event == 'Ok':  # if user closes window or clicks cancel
    #         break
    # window.close()
    gui.popup('This is your generated value: ' + str(number))