# https://github.com/qismib/QuITE/blob/5b98ddc8c8e3cd82d4f9887fc84a70b85ad05b27/programs/real_device/qteleportation_real_device.py
from qiskit import *
from funzioni import *
import matplotlib
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import *
from qiskit.extensions import Initialize
from qiskit.tools.monitor import job_monitor


#loading my IBMQ account
provider = IBMQ.load_account()

#configuaration
Alice_cr0 = ClassicalRegister(1)
Alice_cr1 = ClassicalRegister(1)
Bob_cr = ClassicalRegister(1)
qr = QuantumRegister(3)
circuit = QuantumCircuit(qr, Alice_cr0, Alice_cr1, Bob_cr)



#trasformo il primo qubit in un generico stato |psi> che voglio teletrasportare
#create_a_random_state0(circuit, qr)
create_a_random_state1(circuit, qr)


#creo l'entanglement tra il secondo e terzo qubit
Bell_pair(circuit, qr)

#alice applica le sue porte
Alice_operations(circuit, qr)

 #bob applica l'operazione corretta
Bob_operations_real_device(circuit, qr)

 #verifica
checking(circuit, qr, Bob_cr)

#misure finali
Alice_measurements(circuit, qr, Alice_cr0, Alice_cr1)
circuit.measure(qr[2], Bob_cr)

realQc = provider.get_backend('ibmq_5_yorktown')
job = execute(circuit, backend = realQc, shots = 1000)

job_monitor(job)
reasults = job.result()

plot_histogram(reasults.get_counts(circuit))

circuit.draw(output = 'mpl')


plt.show()
