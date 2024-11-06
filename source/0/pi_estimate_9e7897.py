# https://github.com/jamesjcai/quantum/blob/c3cecfb4deff561ad39b4b97215ae7e80d2fba76/pi_estimate.py
## import the necessary tools for our work
from IPython.display import clear_output
from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plotter
from qiskit.tools.monitor import job_monitor
# Visualisation settings
import seaborn as sns, operator
sns.set_style("dark")

pi = np.pi

## Code for inverse Quantum Fourier Transform
## adapted from Qiskit Textbook at
## qiskit.org/textbook

def qft_dagger(circ_, n_qubits):
    """n-qubit QFTdagger the first n qubits in circ"""
    for qubit in range(int(n_qubits/2)):
        circ_.swap(qubit, n_qubits-qubit-1)
    for j in range(0,n_qubits):
        for m in range(j):
            circ_.cp(-np.pi/float(2**(j-m)), m, j)
        circ_.h(j)



def qpe_pre(circ_, n_qubits):
    circ_.h(range(n_qubits))
    circ_.x(n_qubits)
    for x in reversed(range(n_qubits)):
        for _ in range(2**(n_qubits-1-x)):
            circ_.cp(1, n_qubits-1-x, n_qubits)
            print(f"{n_qubits-1-x} | {n_qubits}")



def run_job(circ, backend, shots=1000, optimization_level=0):
    t_circ = transpile(circ, backend, optimization_level=optimization_level)
    qobj = assemble(t_circ, shots=shots)
    job = backend.run(qobj)
    job_monitor(job)
    return job.result().get_counts()


simulator = Aer.get_backend('aer_simulator')

def get_pi_estimate(n_qubits):
    # create the circuit
    circ = QuantumCircuit(n_qubits + 1, n_qubits)
    # create the input state
    qpe_pre(circ, n_qubits)
    # apply a barrier
    circ.barrier()
    # apply the inverse fourier transform
    qft_dagger(circ, n_qubits)
    # apply  a barrier
    circ.barrier()
    # measure all but the last qubits
    circ.measure(range(n_qubits), range(n_qubits))
    print(circ.draw())
    # run the job and get the results
    counts = run_job(circ, backend=simulator, shots=10000, optimization_level=0)
    # print(counts) 
    # get the count that occurred most frequently
    max_counts_result1 = max(counts, key=counts.get)
    print(f"aaa {max_counts_result1}")

    max_counts_result = int(max_counts_result1, 2)
    print(f"bbb {max_counts_result}")
    # solve for pi from the measured counts
    theta = max_counts_result/2**n_qubits
    return (1./(2*theta))


a=get_pi_estimate(4)
print(f"ccc {a}")


# estimate pi using different numbers of qubits
#nqs = list(range(2,12+1))
#pi_estimates = []
#for nq in nqs:
#    thisnq_pi_estimate = get_pi_estimate(nq)
#    pi_estimates.append(thisnq_pi_estimate)
#    print(f"{nq} qubits, pi ≈ {thisnq_pi_estimate}")



#plotter.plot(nqs, [pi]*len(nqs), '--r')
#plotter.plot(nqs, pi_estimates, '.-', markersize=12)
#plotter.xlim([1.5, 12.5])
#plotter.ylim([1.5, 4.5])
#plotter.legend(['$\pi$', 'estimate of $\pi$'])
#plotter.xlabel('Number of qubits', fontdict={'size':20})
#plotter.ylabel('$\pi$ and estimate of $\pi$', fontdict={'size':20})
#plotter.tick_params(axis='x', labelsize=12)
#plotter.tick_params(axis='y', labelsize=12)
#plotter.show()