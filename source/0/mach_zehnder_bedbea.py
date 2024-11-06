# https://github.com/alexex/Quantum/blob/e17479ca5a741d4e0a176fd0900afb14754c682d/Mach_Zehnder.py
from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit_textbook.tools import array_to_latex
import matplotlib.pyplot as plt

print()
    
print('''1: Beam splitter 1 e rilevatori
2: Beam splitter 1 + specchi e rilevatori
3: Specchio trasparente 1 + specchi + specchio trasparente 2 e rilevatori
4: Specchio trasparente 1 + specchi + beam splitter e rilevatori
5: Completo... Beam splitter 1 + specchi + beam splitter 2 e rilevatori
9: Exit
.''')
 
while True:
    qc = QuantumCircuit(1)
    a = input('Scelta: ')
    if a == '1':
        qc.h(0)
        tt = '1: Beam splitter 1 e rilevatori'
    elif a == '2':
        qc.h(0)
        qc.x(0)
        tt = '2: Beam splitter 1 + specchi e rilevatori'
    elif a == '3':
        qc.i(0)
        qc.x(0)
        qc.i(0)
        tt = '3: Specchio trasparente 1 + specchi + specchio trasparente 2 e rilevatori'
    elif a == '4':
        qc.i(0)
        qc.x(0)
        qc.h(0)
        tt = '4: Specchio trasparente 1 + specchi + beam splitter e rilevatori'
    elif a == '5':
        qc.h(0)
        qc.x(0)
        qc.h(0)
        tt = 'Beam splitter 1 + specchi + beam splitter 2 e rilevatori'
    elif a == '9':
        break
    else:
        print('???????')
        continue

    qc.draw()
    print(qc)
    
    print()
    
    svsim = Aer.get_backend('statevector_simulator')
    qobj = assemble(qc)
    result = svsim.run(qobj).result()
    final_state = result.get_statevector()
    
    isto = result.get_counts()
    plot_histogram(isto)
    plt.title(tt)
    plt.show()
