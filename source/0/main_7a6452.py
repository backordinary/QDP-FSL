# https://github.com/sleepingcat4/QPC/blob/34304b3d3982f2404caf24f3526b8ee114726f48/main.py
import tkinter as tk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

def factorize():
    number = int(number_entry.get()) # get the number from the UI
    q = QuantumRegister(3) # create a quantum register with 3 qubits
    c = ClassicalRegister(3) # create a classical register with 3 bits
    circuit = QuantumCircuit(q, c)
    circuit.h(q[0]) # put the first qubit in the state |+>
    circuit.x(q[1]) # put the second qubit in the state |1>
    circuit.h(q[1]) # put the second qubit in the state |+>
    circuit.u1(3.14, q[2]) # apply a phase shift of pi/4 to the third qubit
    circuit.cx(q[1], q[2]) # apply a controlled-X gate between qubits 1 and 2
    circuit.u1(3.14, q[2]) # apply a phase shift of pi/4 to the third qubit
    circuit.cx(q[0], q[2]) # apply a controlled-X gate between qubits 0 and 2
    circuit.measure(q, c)
    backend = Aer.get_backend('qasm_simulator') # use the qasm simulator
    job = execute(circuit, backend)
    result = job.result()
    counts = result.get_counts()
    factors = []
    for key in counts.keys():
        factors.append(int(key,2))
    output_label.config(text="Factors of "+str(number)+" are "+str(factors))


root = tk.Tk()
root.title("Shor's Algorithm")

number_label = tk.Label(root, text="Enter a number:")
number_label.pack()

number_entry = tk.Entry(root)
number_entry.pack()

factorize_button = tk.Button(root, text="Factorize", command=factorize)
factorize_button.pack()

output_label = tk.Label(root)
output_label.pack()

root.mainloop()

