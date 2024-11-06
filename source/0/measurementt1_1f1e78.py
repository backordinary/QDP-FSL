# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutoriale-QISKit/reference/relaxationAndDecoherence/measurementT1.py
import sys, getpass, os
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../', 'qiskit-sdk-py'))
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
    print('Qconfig loaded from %s.' %Qconfig.__file__) 
except:
    APItoken = getpass.getpass('Please imput your token and hit enter: ', da2a4002660558a35103a600bcbda7fe438cea629a6be98969ea5e367c091b6815e624bd86b6207121bd97fef79c22033318a4402eeafcbd04b021fd80f5a195)
    qx_config = {
        "APItoken": APItoken,
        "url":"https://quantumexperience.ng.bluemix.net/api"}
    print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

import qiskit as qk 
import numpy as np 
from scipy.optimize import curve_fit
from qiskit.tools.qcvv.fitters import exp_fit_fun, osc_fit_fun, plot_coherence

# function for padding with QId gates
def pad_QId(circuit, N, qr):
    # circuit to add to, N = number of QId gates to add, qr = qubit register
    for ii in range(N):
        circuit.barrier(qr)
        circuit.iden(qr)
    return circuit

qk.register(qx_config['APItoken'], qx_config['url'])

# backend and token settings
backend = qk.get_backend('ibmqx4') # the device to run on
shots = 1024 # number of shots in the experiment

# Select qubit whosr T1 is to be measured
qubit = 1

# Creating registers
qr = qk.QuantumRegister(5)
cr = qk.ClassicalRegister(5)

# the delay times are all set in terms of single-qubit gates
# so we need to calculate the time from these parameters

params = backend.parameters['qubits'][qubit]
pulse_length = params['gateTime']['value'] # single-qubit gate time
buffer_length = params['buffer']['value']
unit = params['gateTime']['unit']

steps=10
gates_per_step=120
max_gates=(steps-1)*gates_per_step+1
tot_length=buffer_length+pulse_length
time_per_step=gates_per_step*tot_length
qc_dict={}
for ii in range(steps):
    step_num='step_%s'%(str(ii))
    qc_dict.update({step_num:qk.QuantumCircuit(qr, cr)})
    qc_dict[step_num].x(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num],gates_per_step*ii,qr[qubit])
    qc_dict[step_num].barrier(qr[qubit])
    qc_dict[step_num].measure(qr[qubit], cr[qubit])

circuits=list(qc_dict.values())

# run the program
status = backend.status
if status['available'] == False or status['pending_jobs'] > 10:
    print('Warning: the selected backend appears to be busy or unavailable at present; consider choosing a different on if possible')
t1_job=qk.execute(circuits, backend, shots=shots)

# arrange the data from the run

result_t1 = t1_job.result()
keys_0_1=list(result_t1.get_counts(qc_dict['step_0']).keys()) # get the key of the excited state '00001'

data=np.zeros(len(qc_dict.keys())) # numpy array for data
sigma_data = np.zeros(len(qc_dict.keys()))

# change unit from ns to microseconds
plot_factor = 1
if unit.find('ns')>-1:
    plot_factor=1000
    punit='$\mu$s'
xvals=time_per_step*np.linspace(0,len(qc_dict.keys()),len(qc_dict.keys()))/plot_factor # calculate the time steps in microseconds

for ii,key in enumerate(qc_dict.keys()):
    # get the data in terms of counts for the excited state normalized to the total number of counts
    data[ii]=float(result_t1.get_counts(qc_dict[key])[keys_0_1[1]])/shots
    sigma_data[ii] = np.sqrt(data[ii]*(1-data[ii]))/np.sqrt(shots)

# fit the data to an exponetial 
fitT1, fcov = curve_fit(exp_fit_fun, xvals, data, bounds=([-1,2,0], [1., 500, 1]))
ferr = np.sqrt(np.diag(fcov))

plot_coherence(xvals, data, sigma_data, fitT1, exp_fit_fun, punit, 'T$_1$', qubit)

print("a: " + str(round(fitT1[0],2)) + u" \u00B1 " + str(round(ferr[0],2)))
print("T1: " +str(round(fitT1[1],2)) + " µs" + u"\u00B1 " + str(round(ferr[1],2)) + ' µs')
print("c: " + str(round(fitT1[2],2)) + u" \u00B1 " + str(round(ferr[2],2)))
str(params['T1']['value']) + ' ' + params['T1']['unit']