# https://github.com/jjhorder/qiskit_scripts/blob/49bd12b2740e44adb00edf18b396bfa64d6c2246/measure_bell_state.py
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import noise
import matplotlib.pyplot

#%matplotlib inline     <--- gives "SyntaxError: invalid syntax"
'exec(%matplotlib inline)'

### doesn't work well in jupyter qtconsole when uncommented
### so need to run this line in console separately
#IBMQ.load_account()

# create a blank circuit with two qubits and two bits
qc = QuantumCircuit(2, 2)

# create the Bell state |+>
qc.h(0)
qc.cx(0, 1)

# read qubits to bits
qc.measure(0, 0)
qc.measure(1, 1)
qc.draw(output = 'mpl')

# rum on simulated ideal quantum computer
emulator = Aer.get_backend('qasm_simulator')
job_ideal = execute(qc, emulator, shots = 8192)
hist_ideal = job_ideal.result().get_counts()    # returns a dictionary
hist_ideal['01'] = 0
hist_ideal['10'] = 0
plot_histogram(hist_ideal, title = 'Ideal quantum computation')

# run on simulated noisy quantum computer
provider = IBMQ.get_provider(hub = 'ibm-q')
real_device = provider.get_backend('ibmq_16_melbourne')
properties = real_device.properties()
coupling_map = real_device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(properties)

job_noisy = execute(qc, emulator, shots = 8192, noise_model = noise_model, coupling_map = coupling_map, basis_gates = noise_model.basis_gates)
hist_noisy = job_noisy.result().get_counts()
plot_histogram(hist_noisy, title = 'Noisy quantum computation')
