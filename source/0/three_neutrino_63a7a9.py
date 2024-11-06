# https://github.com/pablocastano03/quantum_computing_and_neutrinos/blob/39e37718fe5b0975b3d30a6aa25c290eafb34e66/three_neutrino.py
from qiskit import Aer, execute, ClassicalRegister, QuantumRegister, QuantumCircuit, IBMQ
from collections.abc import Iterable
import numpy as np

from physicsconstants import OscParams

paper_PMNS_param        = (-0.6031, 7.412,   0.7966,  1.0139, 0.7053, -8.065)
paper_PMNS_dagger_param = (-0.7053, -1.3599, 0.7966, -1.0139, 0.6031, 2.0125)
op = OscParams()

def parse_bool(boolable):
    if boolable in [1, 'y', 'yes', 'True', True, 'Y', '1']:
        return True
    elif boolable in [0, 'n', 'no', 'False', False, 'N', '0']:
        return False
    else:
        print('Cannot parse bool.')
        return None

def initialize_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='outfile', default='three_neutrino')
    parser.add_argument('--backend', dest='backend', default='sim', 
                        help='Which backend to use. Either "sim" or "qc"')
    parser.add_argument('--error_test', dest='error', default='False')
    parser.add_argument('--init_state', default='numu')
    args = parser.parse_args()
    return args


def phi(m2, LoE):
    r'''
    params
    ______
    m2  (float): Mass squared difference [ev^2]
    LoE (float): Baseline over energy [km/GeV]
    
    returns
    _______
    phi (float): 
    '''

    coef = 2.534
    phi  = coef * m2 * LoE
    return phi

class ThreeNuOscillator:

    def __init__(self, init_state):
        self.qreg   = QuantumRegister(2)
        self.creg   = ClassicalRegister(2)
        self.qc     = QuantumCircuit(self.qreg, self.creg)
        if init_state=='nue':
            pass
        elif init_state=='numu':
            self.qc.x(self.qreg[0])
        elif init_state=='nutau':
            self.qc.x(self.qreg[1])
        elif init_state=='nus':
            self.qc.x(self.qreg[0])
            self.qc.x(self.qreg[1])
        else:
            print('init_state %s not recognized. Please reinitialize' % init_state)
        self.counts = None
        

    def apply_rotation(self, param):
        r'''
    
        '''
        alpha, beta, gamma, delta, epsilon, zeta = param
        self.qc.u(beta, 0, 0, self.qreg[0])
        self.qc.u(alpha, 0, 0, self.qreg[1])
        self.qc.cnot(self.qreg[0], self.qreg[1])
        self.qc.u(delta, 0, 0, self.qreg[0])
        self.qc.u(gamma, 0, 0, self.qreg[1])
        self.qc.cnot(self.qreg[0], self.qreg[1])
        self.qc.u(zeta, 0, 0, self.qreg[0])
        self.qc.u(epsilon, 0, 0, self.qreg[1])


    def propoagate(self, LoE, m12=op.deltam12, m13=op.deltam3l):
        r'''

        '''
        self.qc.rz(phi(m12, LoE), self.qreg[0])
        self.qc.rz(phi(m13, LoE), self.qreg[1])

    def measure(self):
        self.qc.measure(self.qreg, self.creg)

if __name__=='__main__':
    
    args = initialize_args()
    
    if args.backend=='qc':
        provider = IBMQ.load_account()
        backend = provider.backends.ibmq_vigo
    elif args.backend=='sim':
        backend = Aer.get_backend('qasm_simulator')
    else:
        print('Backend not recognized. Only "qc" and "sim" supported at this time.')
        quit()
    if parse_bool(args.error):
        PMNS_param        = (0, 0, 0, 0, 0, 0)
        PMNS_dagger_param = (0, 0, 0, 0, 0, 0)
        outfile           = 'error_check_'+args.init_state
    else:
        PMNS_param        = paper_PMNS_param
        PMNS_dagger_param = paper_PMNS_dagger_param
        outfile           = '%s_%s_%s' % (args.outfile, args.init_state, args.backend)
        print(outfile)

    loee = np.linspace(0, 1200, 21)
    
    n = 1024
    results = np.zeros((4,len(loee)))
    for i, LE in enumerate(loee):
        tno = ThreeNuOscillator(args.init_state)
        tno.apply_rotation(PMNS_dagger_param)
        tno.propoagate(LE)
        tno.apply_rotation(PMNS_param)
        tno.measure()
        job = execute(tno.qc, backend, shots=n)
        counts = job.result().get_counts(tno.qc)
        for key in ['00', '01', '10', '11']: # add keys with 0 counts by hand
            if key not in counts.keys():
                counts[key] = 0
        for j, (key, val) in enumerate(sorted(counts.items())):
            print(key)
            results[j, i] = float(val)/n
    np.save(outfile, results)
