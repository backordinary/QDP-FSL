# https://github.com/marcoradaelli/TesiLaureaMagistrale/blob/6d6bc9e9ce74ea92313ebaf8b383092ab1624a0e/qiskit_entropia.py
import numpy as np
import qiskit
from matplotlib import pyplot as plt
from qiskit.tools.visualization import circuit_drawer, plot_histogram
from walks_core import qrw, anello, physics_utilities as phu
from strumenti_analisi import utilita_qiskit as uq

# Questa funzione genera un quantum incrementer.
def genera_incrementer():
    qr = qiskit.QuantumRegister(7, "q")
    incrementer = qiskit.QuantumCircuit(qr, name="incr")
    incrementer.cx(0, 1)
    incrementer.ccx(1, 2, 3)
    incrementer.ccx(3, 4, 5)
    incrementer.cx(5, 6)
    incrementer.ccx(3, 4, 5)
    incrementer.cx(3, 4)
    incrementer.ccx(1, 2, 3)
    incrementer.cx(1, 2)
    incrementer.cx(0, 1)
    incrementer.x(0)

    # Trasformo in un unico gate.
    gate_incrementer = incrementer.to_gate()
    return gate_incrementer


def genera_decrementer():
    qr = qiskit.QuantumRegister(7, "q")
    decrementer = qiskit.QuantumCircuit(qr, name="decr")
    decrementer.x(0)
    decrementer.x(2)
    decrementer.x(4)
    decrementer.cx(0, 1)
    decrementer.ccx(1, 2, 3)
    decrementer.ccx(3, 4, 5)
    decrementer.cx(5, 6)
    decrementer.ccx(3, 4, 5)
    decrementer.x(4)
    decrementer.cx(3, 4)
    decrementer.ccx(1, 2, 3)
    decrementer.x(2)
    decrementer.cx(1, 2)
    decrementer.cx(0, 1)

    # Trasformo in un unico gate.
    gate_decrementer = decrementer.to_gate()
    return gate_decrementer


def prepara_stato_iniziale(stato_coin:int, posizione_iniziale:str):
    qall = qiskit.QuantumRegister(8)
    initializer = qiskit.QuantumCircuit(qall)
    lista_input = list(posizione_iniziale)
    # Correggo per la convenzione di ordine.
    lista_input.reverse()  # Attenzione: l'uso di reverse modifica la lista!
    for i in range(len(posizione_iniziale)):
        if int(lista_input[i]) == 1:
            initializer.x(i * 2 + 1)
    if stato_coin == 1:
        initializer.x(0)

    # Trasformo in un unico gate.
    gate_initializer = initializer.to_gate()
    return gate_initializer


def genera_operatore_passo():
    qall = qiskit.QuantumRegister(8)
    step_op = qiskit.QuantumCircuit(qall)

    # Genero gli operatori di incremento e decremento e li controllo.
    ctrl_incrementer = genera_incrementer().control(1)
    ctrl_decrementer = genera_decrementer().control(1)

    # Costruisco l'operatore di passo.
    step_op.h(0)
    step_op.append(ctrl_incrementer, qall)
    step_op.x(0)
    step_op.append(ctrl_decrementer, qall)
    step_op.x(0)

    # Trasformo in un unico gate.
    gate_step_op = step_op.to_gate()
    return gate_step_op


def esegui_passi_quantum(numero_passi: int):
    qr = qiskit.QuantumRegister(8)
    cr = qiskit.ClassicalRegister(4)
    circuit = qiskit.QuantumCircuit(qr, cr)

    # Preparo lo stato iniziale.
    coin_iniziale = 0
    posizione_iniziale = "0000"
    initializer = prepara_stato_iniziale(coin_iniziale, posizione_iniziale)

    operatore_passo = genera_operatore_passo()

    # Creo il circuito.
    circuit.append(initializer, qr)
    for passo in range(numero_passi):
        circuit.append(operatore_passo, qr)
    # Inserisco le misure alla fine.
    circuit.measure(1,0)
    circuit.measure(3,1)
    circuit.measure(5,2)
    circuit.measure(7,3)

    backend = qiskit.Aer.get_backend("aer_simulator")
    numero_run = 1000
    job = qiskit.execute(circuit, backend, shots=numero_run)
    result = job.result()
    outcome = result.get_counts().int_outcomes()
    ddp = uq.rappresenta_risultati_tramite_array(outcome, 16, numero_run)

    return ddp



a = anello.anello(16)
w = qrw.walker(a,0,[1,0],"hadamard")
vett_entropie_mie = []
vett_entropie_ibm = []

numero_massimo_passi = 60
for passo in range(numero_massimo_passi):
    ddp_mia, appo = w.ottieni_distribuzione_probabilita()
    ddp_ibm = esegui_passi_quantum(passo)

    # Calcolo le entropie di Shannon.
    entropia_mia = phu.entropia_shannon(ddp_mia)
    entropia_ibm = phu.entropia_shannon(ddp_ibm)

    vett_entropie_mie.append(entropia_mia)
    vett_entropie_ibm.append(entropia_ibm)

    w.passo()
    print("Eseguito passo ", passo, "::: entropia mia ", entropia_mia, ";     entropia IBM ", entropia_ibm)

plt.plot(vett_entropie_mie, label="Ideal")
plt.plot(vett_entropie_ibm, label="ibm_qasm")
plt.hlines(y=np.log2(8), xmin=0, xmax=numero_massimo_passi, label="Maximal entropy")
plt.xlabel("Steps")
plt.ylabel("Shannon entropy")
plt.suptitle("Shannon entropy on circuit-like architecture")
plt.title("Number of points in the cycle: 16; Run on ibm_qasm: 1000")
plt.legend()
plt.show()