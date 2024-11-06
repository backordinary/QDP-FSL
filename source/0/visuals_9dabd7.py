# https://github.com/cpbunker/learn_qiskit/blob/a1c81db8f61f1f00cc710b6f05ce17fdb623e6f2/mining/visuals.py
'''
https://github.com/cpbunker/learn_qiskit
'''

import qiskit
from qiskit import QuantumCircuit
from qiskit import quantum_info as qinfo

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#try:  plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

######################################################################
#### bloch sphere visualization

def bloch_qubits(state: qinfo.Statevector, my_title = '') -> None:
    '''
    Take a quantum_info.Statevector object and plot the state of the qubitith
    qubit on the bloch sphere
    '''
    assert(isinstance(state, qinfo.Statevector));

    state.draw(output = 'bloch', title = my_title, reverse_bits = True);
    plt.show();

    return None;


def bloch_state(state: qinfo.Statevector, vec0: str, vec1: str) -> None:
    '''
    Draw a (single or multiqubit) state on the bloch sphere with poles vec0, vec1
    '''
    from qiskit.visualization import plot_bloch_vector
    if(not isinstance(state, qinfo.Statevector)): raise TypeError;
    statedict = state.to_dict();
    assert(vec0 in statedict and vec1 in statedict ); # allowed poles

    # get coefs
    coef0, coef1 = statedict[vec0], statedict[vec1];

    # convert to bloch sphere coordinates (see
    # https://en.wikipedia.org/wiki/Bloch_sphere#Plotting_pure_two-spinor_states_through_stereographic_projection)
    u = coef1/coef0; # complex number
    ux, uy = u.real, u.imag; # break into ux + i uy form
    px = 2*ux/(1 + ux*ux + uy*uy); # p is the bloch sphere point
    py = 2*uy/(1 + ux*ux + uy*uy);
    pz = (1 - ux*ux - uy*uy)/(1 + ux*ux + uy*uy);

    # plot using qiskit
    plot_bloch_vector([px, py, pz], title = '|0> = '+vec0+', |1> = '+vec1);
    plt.show();

######################################################################
#### circuit visualization

def circuit_counts(qc: QuantumCircuit, shots = 1024) -> None:
    '''
    Do a quantum simulation of myqc and plot histogram of the results

    if qc has clbits, assume there is a measurment scheme manually set up
    if not we will just measure all the qubits to new clbits
    '''
    from qiskit import Aer
    from qiskit.providers.aer import AerSimulator
    from qiskit.visualization import plot_histogram
    qc = qc.copy();
    if(not isinstance(qc, QuantumCircuit)): raise TypeError;

    # custom measurement
    if(len(qc.clbits)):
        pass; # no need to do anything

    else: # measure onto new clbits measure_all
        qc.measure_all(add_bits = True);  

    # run the job
    if True:
        mybackend = Aer.get_backend('qasm_simulator');
        result = qiskit.execute(qc, backend = mybackend, shots = shots).result();
        counts = result.get_counts();

    else:
        job = AerSimulator().run(qc); # physically runs the qc
        counts = job.result().get_counts();

    # show
    plot_histogram(counts);
    plt.show();
    return None;

######################################################################
#### test code
    
if(__name__ == "__main__"):

    # null state
    null = qinfo.Statevector.from_label('00');

    # phi + bell state
    bell_qc = QuantumCircuit(2);
    bell_qc.h(1);
    bell_qc.cnot(0,1);
    print(bell_qc.draw());
    phi_plus = null.evolve(bell_qc);
    print(phi_plus.to_dict())

    # draw qubits
    bloch_qubits(phi_plus);

    # draw state
    bloch_state(phi_plus, '00', '10');
