# https://github.com/patevs/quantum-pi/blob/2c01e25c5bcb7cab3e4c2098d7b071ec2f7e8cf5/src/quantum_pi/main.py
#!/usr/bin/env python

# * ---------------------- * #
# * src/quantum_pi/main.py * #
# * ---------------------- * #

"""
This file that can serves as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        quantum-pi = quantum_pi.main:run

Then run `python setup.py install` which will install the command `quantum-pi`
inside your current environment.
"""

# * ------- * #
# * IMPORTS * #
# * ------- * #

import argparse
import sys
import logging

from quantum_pi import __version__

from IPython.display import clear_output
from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plotter
from qiskit.tools.monitor import job_monitor

import os
from dotenv import load_dotenv
load_dotenv()

# Visualisation settings
import seaborn as sns, operator
sns.set_style("dark")


# * --------- * #
# * CONSTANTS * #
# * --------- * #

__author__ = "Patrick Evans"
__copyright__ = "Patrick Evans"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

pi = np.pi

IBM_QUANTUM_KEY = os.getenv("IBM_QUANTUM_KEY")


# * --------- * #
# * FUNCTIONS * #
# * --------- * #

## Code for inverse Quantum Fourier Transform
## adapted from Qiskit Textbook at
## qiskit.org/textbook
def qft_dagger(circ_, n_qubits):
    """n-qubit QFTdagger the first n qubits in circ"""
    for qubit in range(int(n_qubits/2)):
        circ_.swap(qubit, n_qubits-qubit-1)
    for j in range(0,n_qubits):
        for m in range(j):
            circ_.cu1(-np.pi/float(2**(j-m)), m, j)
        circ_.h(j)


## Code for initial state of Quantum Phase Estimation
## adapted from Qiskit Textbook at qiskit.org/textbook
## Note that the starting state is created by applying
## H on the first n_qubits, and setting the last qubit to |psi> = |1>

def qpe_pre(circ_, n_qubits):
    circ_.h(range(n_qubits))
    circ_.x(n_qubits)

    for x in reversed(range(n_qubits)):
        for _ in range(2**(n_qubits-1-x)):
            circ_.cu1(1, n_qubits-1-x, n_qubits)


## Run a Qiskit job on either hardware or simulators

def run_job(circ_, backend_, shots_=1000, optimization_level_=0):
    job = execute(circ_, backend=backend_, shots=shots_, optimization_level=optimization_level_)
    job_monitor(job)
    return job.result().get_counts(circ_)


## Load your IBMQ account if
## you'd like to use the cloud simulator or real quantum devices

IBMQ.save_account(IBM_QUANTUM_KEY, overwrite=True)
IBMQ.load_account()
my_provider = IBMQ.get_provider()
simulator_cloud = my_provider.get_backend('ibmq_qasm_simulator')
simulator = Aer.get_backend('qasm_simulator')
# device = my_provider.get_backend('ibmq_16_melbourne')
# device = my_provider.get_backend('ibm_auckland')
device = my_provider.get_backend('ibmq_lima')


## Function to estimate pi
## Summary: using the notation in the Qiskit textbook (qiskit.org/textbook),
## do quantum phase estimation with the operator U = u1(theta) and |psi> = |1>
## such that u1(theta)|1> = exp(2 x pi x i x theta)|1>
## By setting theta = 1 radian, we can solve for pi
## using 2^n x 1 radian = most frequently measured count = 2 x pi

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

    # if n_qubits < 10:
    #     circ.draw(output='mpl').savefig('piday-code-output/'+str(n_qubits)+'_qubit_circuit.png')

    # run the job and get the results
    counts = run_job(circ, backend_=simulator, shots_=10000, optimization_level_=0)
    # print(counts)

    # get the count that occurred most frequently
    max_counts_result = max(counts, key=counts.get)
    max_counts_result = int(max_counts_result, 2)

    # solve for pi from the measured counts
    theta = max_counts_result/2**n_qubits
    return (1./(2*theta))


def plot_results(nqs, pi_estimates):
	# Plot the results
	plotter.plot(nqs, [pi]*len(nqs), '--r')
	plotter.plot(nqs, pi_estimates, '.-', markersize=12)
	plotter.xlim([1.5, 12.5])
	plotter.ylim([1.5, 4.5])
	plotter.legend(['$\pi$', 'estimate of $\pi$'])
	plotter.xlabel('Number of qubits', fontdict={'size':20})
	plotter.ylabel('$\pi$ and estimate of $\pi$', fontdict={'size':20})
	plotter.tick_params(axis='x', labelsize=12)
	plotter.tick_params(axis='y', labelsize=12)
	plotter.show()


def parse_args(args):
    """Parse command line parameters

    Args:
    	args ([str]): command line parameters as list of strings

    Returns:
    	:obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Estimating Pi Using Quantum Phase Estimation Algorithm")
    parser.add_argument(
        "--version",
        action="version",
        version="quantum-pi {ver}".format(ver=__version__))
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
    	loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
    	args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    # main(sys.argv[1:])
    # estimate pi using different numbers of qubits
    nqs = list(range(2,12+1))
    pi_estimates = []
    for nq in nqs:
        thisnq_pi_estimate = get_pi_estimate(nq)
        pi_estimates.append(thisnq_pi_estimate)
        print(f"{nq} qubits, pi ≈ {thisnq_pi_estimate}")
    plot_results(nqs, pi_estimates)


if __name__ == "__main__":
    run()

# EOF #
