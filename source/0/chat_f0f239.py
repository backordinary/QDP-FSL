# https://github.com/sleepingcat4/Quantum-Chat-app/blob/0e839bb3012538b9f7583a31169ac07cbc5f37c2/chat.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from tkinter import Tk, Label, Button, Entry
import time

def send_message():
    # Get the message from the text box
    message = message_entry.get()

    # Use Aer's qasm_simulator to execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    result = job.result()

    # Use the result of the measurement to send a message
    if result.get_counts(qc) == {'0': 1}:
        received_message_label.config(text="Received message: " + message)
    else:
        received_message_label.config(text="Received message: " + message)

# Create a quantum circuit with one qubit and one classical bit
q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)

# Use the Hadamard gate to put the qubit in a superposition
qc.h(q[0])

# Measure the qubit and store the result in the classical bit
qc.measure(q, c)

# Create the UI for the sending user
root = Tk()
root.title("Quantum Chat App - Sending User")

message_entry = Entry(root)
message_entry.pack()

send_button = Button(root, text="Send message", command=send_message)
send_button.pack()

message_label = Label(root)
message_label.pack()

# Create the UI for the receiving user
received_message_label = Label(root, text="Waiting for message...")
received_message_label.pack()

# Continuously check for new messages
while True:
    root.update()
    time.sleep(0.1)
