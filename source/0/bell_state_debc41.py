# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/bell_state.py
from qiskit import*
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt



quantum_register = QuantumRegister(2)   # Linhas do circuito / numeros de Qbit no circuito
""" Para medir, em mecanica quântica, ao medir um Qbit nós destruimos a informação daquele estado
 e armazenamos ela em um bit clássico """
classic_register = ClassicalRegister(2) 

first_circuit = QuantumCircuit(quantum_register, classic_register) # Constrói o circuito

first_circuit.draw(output = 'mpl') # Desenha o circuito, funciona no app da IBM

first_circuit.h(quantum_register[0])  # Aplicando a primeira gate no primeira linha, gate hadarmat 
# Aplicando a porta CNOT
###
first_circuit.draw(output = 'mpl')
###
first_circuit.cx(quantum_register[0], quantum_register[1]) #CNOT faz operação tensorial entre o Qbit 
#de controle na linha zero, e o outro é o Qbit alvo

###
first_circuit.draw(output = 'mpl')
###
first_circuit.measure(quantum_register, classic_register) # para extrair a medida 
###
first_circuit.draw(output = 'mpl')
###
simulator = QasmSimulator() # Simulador que vai realizar os calculos para nós 

result = execute(first_circuit, backend= simulator).result()

counts = result.get_counts(first_circuit)

first_circuit.draw(output='mpl')

plot_histogram(counts)
plt.ylabel(counts)
plt.show()

"""
IBMQ.load_account()
host = IBMQ.get_provider('ibm-q')
quantum_computer = host.get_backend('ibmq_belem')
result_qcomputer = execute(first_circuit, backend= quantum_computer)

job_monitor(result_qcomputer)

result = result_qcomputer.result()

plot_histogram(result.get_counts(first_circuit))
plt.ylabel(counts)
plt.show()
"""