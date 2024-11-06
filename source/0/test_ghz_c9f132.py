# https://github.com/FredericSauv/QuantumSimulation/blob/0be68edfa33fb570d52e31be1d0b277a611b9ba4/Batch/qiskit/test_GHZ.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:24:40 2020

@author: form Kiran
"""

from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer, transpile
from qiskit.providers.aer.noise import NoiseModel
import numpy as np




simulator = Aer.get_backend('qasm_simulator')
MEAS_DEFAULT = ['xxx', '1zz', 'z1z', 'zz1', 'yyx', 'xyy', 'yxy']
MEAS_WEIGHTS = np.array([1., 1., 1., 1., -1., -1., -1.])/4.0
NB_SHOTS_DEFAULT = 128
pi = np.pi
x_opt = np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
x_loc = np.array([1., 0., 4., 0., 3., 0.]) * np.pi/2



def create_circ(params = np.zeros(6)):
    c = QuantumCircuit(3)
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    c.cnot(0,2) 
    c.cnot(1,2) 
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    return c
   

def append_measurements(circ, measurements):
    """ Assumes creator returns an instance of the relevant circuit"""
    num_classical = len(measurements.replace('1',''))
    if num_classical > 0:
        cr = ClassicalRegister(num_classical, 'c')
        circ.add_register(cr)
    ct_m = 0
    ct_q = 0
    for basis in measurements:
        if basis == 'z':
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'x':
            circ.u3(pi/2, 0, 0, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'y':
            circ.u3(pi/2, -pi/2, -pi/2, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == '1':
            pass
        ct_q+=1
    return circ


def gen_meas_circuits(creator, meas_settings, params):
    """ Return a list of measurable circuit with same parameters but with 
    different measurement settings"""
    c_list = [append_measurements(creator(params), m)for m in meas_settings]
    return c_list

def freq_even(results):
    """ Frequency of +1 eigen values: result 0(1) corresponds to a -1(+1) eigen 
    state. If even number of 0 then +1
    """
    nb_odd, nb_even = 0, 0
    for k, v in results.items():
        nb_even += v * (k.count('0')%2 == 0)
        nb_odd += v * (k.count('0')%2)
    return nb_even / (nb_odd + nb_even)

def F(experimental_params, flags = False, shots = NB_SHOTS_DEFAULT, 
      meas_settings = MEAS_DEFAULT, meas_weights = MEAS_WEIGHTS, 
      noise_model = NoiseModel()):
    """ Main function: take parameters and return an estimation of the fidelity 
    Different possible behaviors:
        +
        + if shots is an integer assign the same number of shots to each meas setting
                      a list assign each entry to a measurement setting
        + if meas_weights is none, returns the frequency associated to each 
          meas_settings, else returns a dot product of weights.frequencies 

    """
    if type(meas_settings) is not list: meas_settings = [meas_settings]
    if type(shots) == int: shots = [shots] * len(meas_settings)
    circs = gen_meas_circuits(create_circ, meas_settings, experimental_params)
    qjobs = [transpile(c, optimization_level=2) for c in circs]

    submitted = []
    for jj, ss in zip(qjobs, shots):
        submitted.append(execute(jj, simulator, shots = ss, noise_model = noise_model))
        if flags:
            print('submitted')

    current_status = str([str(jj.status()) for jj in submitted])
    if flags: print(current_status)
    ###### Uncomment following section to run on IBMQ
    #while 'RUN' in current_status or 'QUE' in current_status: # can optimise to only look at one at a time
    #    print(current_status)
    #    time.sleep(10)
    #    current_status = str([str(jj.status()) for jj in submitted])

    measurement_results = []
    for j in submitted:
        if flags: print(j.result().get_counts())
        measurement_results.append(freq_even(j.result().get_counts()))

    #print(measurement_results)
    if meas_weights is None:
        res = measurement_results
    else:
        res = np.dot(measurement_results, meas_weights)
    #print('Fidelity = {}'.format(str(Fidelity)))
    return np.squeeze(res)


if __name__ == '__main__':    

    print(F(x_loc)) #~0.5
    print(F(x_opt)) # 1.0
    

