# https://github.com/omsotnikov/bit-strings-dissimilarity/blob/c975b3ad613b2d62bcc155d7de0e3b3adcca3c56/dicke.py
from argparse import ArgumentParser
from qiskit import Aer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np


def get_parameters():

    p = ArgumentParser()

    p.add_argument('--qbits', type=int, default=16, help='Total nuber of qbits used in calculation (default: 16).')
    p.add_argument('-n', '--niter', type=int, default=8192, help='Number of generated configurations (default: 8192).')
    p.add_argument('-D', type=int, default=1, help='D parameter of Dicke state (default: 1).')
    p.add_argument('--basis', default='z', choices=('z', 'random'), help='Measurement basis (default: z).')

    p = p.parse_args()
    
    if p.qbits < 1:
        raise ValueError("Total number of qbits should be greater than zero.")

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

    # output file name template
    tmpl = 'Dicke'
    tmpl += '.{0}'.format(p.basis)
    tmpl += '.D={0}.dat'.format(p.D)

    # calculating statevector
    state = np.zeros((2**p.qbits), dtype=complex)
    for i in range(2**p.qbits):
        counts1 = 0
        for ii in range(p.qbits):
            if i & (1<<ii):
                counts1 += 1
        if counts1 == p.D:
            state[i] = complex(1, 0)

    state = state / np.sqrt(np.sum(state**2))

    # create and initialize corresponding quantum circuit
    q = QuantumRegister(p.qbits)
    c = ClassicalRegister(p.qbits)
    dicke = QuantumCircuit(q, c)
    dicke.initialize(np.conj(state), [q[i] for i in range(p.qbits-1,-1,-1)])

    # generate random basis if chosen
    if p.basis == 'random':
        th, ph, la = get_random_basis(p.niter)

    with open(tmpl, 'w') as ffz:

        for it in range(p.niter):
            
            circ = dicke.copy()

            if p.basis == 'random':
                for i in range(p.qbits):
                    circ.u(th[it], ph[it], la[it], q[i])

            circ.barrier()

            for i in range(p.qbits):
                circ.measure(q[i], c[i])

            job = execute(circ, backend=simulator, shots=1)

            result = job.result()
            counts = result.get_counts(circ)

            for bits in counts:
                ffz.write(bits)


if __name__ == '__main__':
    main()
