# https://github.com/omsotnikov/bit-strings-dissimilarity/blob/c975b3ad613b2d62bcc155d7de0e3b3adcca3c56/cats.py
from qiskit import Aer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit
from argparse import ArgumentParser
import numpy as np


def get_parameters():

    p = ArgumentParser()

    p.add_argument('--qbits', type=int, default=16, help='Total nuber of qbits used in calculation (default: 16).')
    p.add_argument('-n', '--niter', type=int, default=8192, help='Number of generated configurations (default: 8192).')
    p.add_argument('--theta', nargs='*', type=float, help='Theta angle(s) of a cat state(s) (default: angles used in paper).')
    p.add_argument('--basis', default='z', choices=('z', 'random'), help='Measurement basis (default: z).')

    p = p.parse_args()

    if p.qbits < 2:
        raise ValueError("Total number of qbits should be greater than 1 for a cat state.")

    return p


def get_random_basis(niter):

    th = np.ndarray((niter,), dtype=float)
    ph = np.ndarray((niter,), dtype=float)
    la = np.ndarray((niter,), dtype=float)

    total = 0
    while total != niter:
        t, p, l = np.random.random((3,))
        t = np.arccos(2*t-1)
        p = 2*np.pi*p
        l = 2*np.pi*l
        if (t <= np.pi/2 and t >= 0 and p <= np.pi/2 and p >= 0 and l <= np.pi/2 and l >= 0):
            th[total] = t
            ph[total] = p
            la[total] = l
            total += 1

    return th, ph, la


def main():

    p = get_parameters()

    simulator = Aer.get_backend('qasm_simulator')

    # get thate angles from command line, otherwise use angles from paper
    theta_angles = p.theta if p.theta else [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]

    # output file name template
    tmpl = 'cats'
    tmpl += '.{0}'.format(p.basis)
    tmpl += '.theta={0:4.3f}.dat'
    
    for theta in theta_angles:

        # generate random basis if chosen
        if p.basis == 'random':
            th, ph, la = get_random_basis(p.niter)

        with open(tmpl.format(theta), 'w') as ffz:

            for it in range(p.niter):

                circ = QuantumCircuit(p.qbits, p.qbits)

                circ.u(theta, 0, 0, 0)
                for i in range(p.qbits-1):
                    circ.cx(i, i+1)

                if p.basis == 'random':
                    for i in range(p.qbits):
                        circ.u(th[it], ph[it], la[it], i)

                circ.barrier()

                for i in range(p.qbits):
                    circ.measure(i, i)

                job = execute(circ, backend=simulator, shots=1)

                result = job.result()
                counts = result.get_counts(circ)

                for bits in counts:
                    ffz.write(bits)


if __name__ == '__main__':
    main()

