# https://github.com/notlosca/quantum_computing/blob/7a6e7d7dbc17cecef6fa317d6092f68ac6191761/codes/codice_HA.py



from qiskit import *
from qiskit import IBMQ
from qiskit.visualization import plot_histogram
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")

qr = QuantumRegister(4)
cr = ClassicalRegister(4)

qcircuit = QuantumCircuit(qr, cr)

get_ipython().run_line_magic('matplotlib', 'inline')



#half adder circuit to add 1 and 1
qcircuit.x(qr[0])
qcircuit.x(qr[1])
#qcircuit.h(qr[2])
#qcircuit.cx(qr[0], qr[1])
qcircuit.ccx(qr[0], qr[1], qr[3]) #controllo, controllo, target
qcircuit.cx(qr[1], qr[2]) #controllo, target
qcircuit.cx(qr[0], qr[2])
qcircuit.measure(qr, cr)
qcircuit.draw(output = 'mpl')


#simulate it in local
simulator = Aer.get_backend('qasm_simulator')
result = execute(qcircuit, backend = simulator).result()
#from qiskit.tools.visualization import plot_histogram
plot_histogram(result.get_counts(qcircuit))



#load the IBM account to launch the circuit on a quantum computer
IBMQ.load_account() #load the account in order to execute the code on the quantum computer


#select the backend
provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_burlington')
job = execute(qcircuit, backend = qcomp, shots = 8192)
from qiskit.tools.monitor import job_monitor
job_monitor(job)



#here we can see how the circuit is transpiled on the quantum computer
from qiskit.visualization import plot_circuit_layout

qc_transpiled = transpile(qcircuit, backend = qcomp, optimization_level=3)
plot_circuit_layout(qc_transpiled, qcomp, view = 'virtual')




#and the error map, thus we can choose if continue with this backend or change it
from qiskit.visualization import plot_error_map
plot_error_map(qcomp) #we plot the error map of the backend selected



#to visualise results

qresult = job.result()
plot_histogram(qresult.get_counts(qcircuit))



#interactive plot

from qiskit.visualization import iplot_histogram
iplot_histogram(qresult.get_counts(qcircuit))







