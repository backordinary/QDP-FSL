# https://github.com/mousepad01/qiskit_POCs/blob/557335b1c3282a71a53fa1f1bc097982b5a1a604/src.py
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info import random_statevector
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import XGate

backend_statevector = Aer.get_backend('statevector_simulator')
backend_sim = Aer.get_backend('qasm_simulator')

with open('t.txt', "r") as token_f:
    TOKEN = token_f.read()

# ===========================

# convert "endianess" of qubit order

def qiskit_tensor_conversion(t: np.array, n: int):

    def _conv(x: int):

        x_ = 0
        for i in range(n):
            x_ |= ((x >> i) & 0x01) << (n - i - 1) 
    
        return x_

    if type(t) == list:
        t = np.array(t)
    
    t_ = deepcopy(t)

    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t_[i][j] = t[_conv(i)][_conv(j)]

    return t_

# simulation for state vector

def get_statevector(circuit: QuantumCircuit):
    return backend_statevector.run(circuit).result().get_statevector(circuit, decimals=2)

def plot_statevector(circuit: QuantumCircuit):
    plot_state_city(get_statevector(circuit))

# simulation for measurement outputs

def simulate_runs(circuit: QuantumCircuit, runs: int = 1024):
    return backend_sim.run(transpile(circuit, backend_sim), shots=runs).result().get_counts()

def plot_simulated_runs(circuit: QuantumCircuit, runs: int = 1024):
    plot_histogram(simulate_runs(circuit, runs))

# real run on IBM 5 qubit quantum computers (measurement outputs)

def real_runs(circuit: QuantumCircuit, runs: int = 1024):

    IBMQ.enable_account(TOKEN)

    provider = IBMQ.get_provider(hub='ibm-q')
    backend_q = least_busy(provider.backends(filters = lambda x: x.configuration().n_qubits == 5 
                                                        and not x.configuration().simulator 
                                                        and x.status().operational == True))

    return backend_q.run(transpile(circuit, backend_q, optimization_level=3), shots=runs).result().get_counts()

def plot_runs(circuit: QuantumCircuit, runs: int = 1024):
    plot_histogram(real_runs(circuit, runs))
    
# ===========================

def example():
    
    c = QuantumCircuit(3, 3)
    
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)

    mes = QuantumCircuit(3, 3)
    mes.measure(range(3), range(3))

    cmes = c.compose(mes)
    
    plot_statevector(c)
    plot_simulated_runs(cmes)

    plt.show()

def superdense_coding_v0(a, b):

    a = [0, 1] if a == 1 else [1, 0]
    b = [0, 1] if b == 1 else [1, 0]
    
    qc_init = QuantumCircuit(4, 2)
    qc_alicebob = QuantumCircuit(4, 2)
    
    qc_init.h(2)
    qc_init.cx(2, 3)
    qc_init.initialize(a, 0)
    qc_init.initialize(b, 1)

    qc_alicebob.cx(1, 2)
    qc_alicebob.cz(0, 2)
    qc_alicebob.cx(2, 3)
    qc_alicebob.h(2)

    qc_alicebob.measure(2, 0)
    qc_alicebob.measure(3, 1)

    superdense_coding_qc = qc_init.compose(qc_alicebob)

    #superdense_coding_qc.draw('mpl')
    plot_simulated_runs(superdense_coding_qc)

    plt.show()

def superdense_coding(a, b):

    def _init_bell_pair():

        bell_state = QuantumCircuit(2)
        bell_state.h(0)
        bell_state.cx(0, 1)

        return bell_state

    def _encode(qc, qubit, b0, b1):

        if b0 not in [0, 1] or b1 not in [0, 1]:
            raise RuntimeError(f'invalid bytes {b0} {b1}')
        
        if b0 == 1:
            qc.x(qubit)

        if b1 == 1:
            qc.z(qubit)
        
        return qc

    def _decode(qc):

        qc.cx(0, 1)
        qc.h(0)
    
        qc.measure_all()
        return qc

    bell_pair = _init_bell_pair()
    bell_pair = _encode(bell_pair, 0, a, b)
    bell_pair = _decode(bell_pair)

    plot_simulated_runs(bell_pair)
    bell_pair.draw('mpl')

    plt.show()

def superdense_coding_real(a, b):

    def _init_bell_pair():

        bell_state = QuantumCircuit(2)
        bell_state.h(0)
        bell_state.cx(0, 1)

        return bell_state

    def _encode(qc, qubit, b0, b1):

        if b0 not in [0, 1] or b1 not in [0, 1]:
            raise RuntimeError(f'invalid bytes {b0} {b1}')
        
        if b0 == 1:
            qc.x(qubit)

        if b1 == 1:
            qc.z(qubit)
        
        return qc

    def _decode(qc):

        qc.cx(0, 1)
        qc.h(0)
    
        qc.measure_all()
        return qc

    bell_pair = _init_bell_pair()
    bell_pair = _encode(bell_pair, 0, a, b)
    bell_pair = _decode(bell_pair)

    plot_runs(bell_pair)
    bell_pair.draw('mpl')

    plt.show()

def quantum_teleportation(p0, p1):

    qc = QuantumRegister(3)
    a, b, psi_measured = ClassicalRegister(1), ClassicalRegister(1), ClassicalRegister(1)
    qc_teleportation = QuantumCircuit(qc, a, b, psi_measured)

    def _init_psi():

        qc_teleportation.initialize([p0, p1], 0)

    def _init_bell_state():

        qc_teleportation.h(1)
        qc_teleportation.cx(1, 2)

        qc_teleportation.barrier()

    def _entangle_psi():

        qc_teleportation.cx(0, 1)
        qc_teleportation.h(0)

        qc_teleportation.barrier()

    def _measure():

        qc_teleportation.measure(0, a)
        qc_teleportation.measure(1, b)

        qc_teleportation.barrier()

    def _apply_xz():

        qc_teleportation.x(2).c_if(b, 1)
        qc_teleportation.z(2).c_if(a, 1)

        qc_teleportation.barrier()

    def _measure_psi():

        qc_teleportation.measure(2, psi_measured)

    _init_psi()
    _init_bell_state()
    _entangle_psi()
    _measure()
    _apply_xz()
    _measure_psi()

    qc_teleportation.draw('mpl')

    psi_cnts = {'0': 0, '1': 0}

    cnts = simulate_runs(qc_teleportation)
    for state, fr in cnts.items():
        psi_cnts[state[0]] += fr

    plot_histogram(psi_cnts)

    plt.show()

def equiv_test():
    
    def _test(a, b, c, d):

        q0_init_state = random_statevector(2)
        q1_init_state = random_statevector(2)

        qc_fst = QuantumCircuit(2)
        qc_fst.initialize(q0_init_state, 0)
        qc_fst.initialize(q1_init_state, 0)

        qc_fst.cx(0, 1)
        
        if a == 1:
            qc_fst.x(0)

        if b == 1:
            qc_fst.z(0)

        if c == 1:
            qc_fst.x(1)

        if d == 1:
            qc_fst.z(1)

        qc_fst.cx(0, 1)

        qc_fst.barrier()
        qc_fst.measure_all()

        qc_snd = QuantumCircuit(2)
        qc_snd.initialize(q0_init_state, 0)
        qc_snd.initialize(q1_init_state, 0)

        a_ = a
        d_ = d
        c_ = (1 - a) * c + a * (1 - c)
        b_ = (1 - d) * b + d * (1 - b)

        if a_ == 1:
            qc_snd.x(0)

        if b_ == 1:
            qc_snd.z(0)

        if c_ == 1:
            qc_snd.x(1)

        if d_ == 1:
            qc_snd.z(1)

        qc_snd.barrier()
        qc_snd.measure_all()

        plot_simulated_runs(qc_fst)
        plot_simulated_runs(qc_snd)

        plt.show()

    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    
                    _test(a, b, c, d)

def deutsch_josza():
    '''
        n = 4
        uses random generated functions that are either constant or balanced
    '''

    # build the Uf operator

    n = 4

    ct = random.randint(0, 1)

    Uf = [[0 for _ in range(1 << (n + 1))] for _ in range(1 << (n + 1))]

    if random.randint(0, 1) == 1:
        constant = True

        for x in range(1 << (n + 1)):
            Uf[x][x] = 1

    else:
        constant = False
        
        f = {x: (ct if x < (1 << (n - 1)) else 1 - ct) for x in range(1 << n)}
        
        for x in range((1 << n) - 1):
            x_ = random.randint(x + 1, (1 << n) - 1)

            aux = f[x]
            f[x] = f[x_]
            f[x_] = aux

        #print(f)

        cnt_ = 0
        for x in range(1 << n):
            if f[x] == 1:
                cnt_ += 1

        assert(cnt_ == (1 << (n - 1)))

        for x in range(1 << (n + 1)):
            
            Uf[x ^ f[x >> 1]][x] = 1

    print(f'function is constant? {constant}')

    #print(Uf)

    #q_uf = Operator(Uf)
    #print(q_uf)

    Uf = qiskit_tensor_conversion(Uf, n + 1)

    #print(Uf)

    q_uf = Operator(Uf)
    #print(q_uf)
    
    qc = QuantumCircuit(n + 1, n)

    for x in range(n):
        qc.initialize([1, 0], x)
        qc.h(x)

    qc.initialize([0, 1], n)
    qc.h(n)
    
    qc.append(q_uf, [x for x in range(n + 1)])

    for x in range(n):
        qc.h(x)
        qc.measure(x, x)

    qc.draw('mpl')
    plot_simulated_runs(qc, runs=128)
    #print(get_statevector(qc))

    plt.show()

def simon_alg():
    '''
        n = 4
        probabilistic version (ends after n + 3 runs)
        it only checks for orthogonality of the returned values wrt s
    '''

    # build the Uf operator

    n = 4

    s = random.randint(1, (1 << n) - 1)

    f_values = [x for x in range(1 << n)]

    f = {x: None for x in range(1 << n)}
    for x in range(1 << n):

        if f[x] is None:

            f[x] = random.choice(f_values)
            f[x ^ s] = f[x]

            f_values.remove(f[x])

    print(f"secret {s}, f {f}")

    Uf = [[0 for _ in range(1 << (n * 2))] for _ in range(1 << (n * 2))]

    for k in range(1 << (n * 2)):
        Uf[k ^ f[k >> n]][k] = 1

    Uf = qiskit_tensor_conversion(Uf, n * 2)
    q_uf = Operator(Uf)

    qc = QuantumCircuit(n * 2, n * 2)

    for x in range(n * 2):
        qc.initialize([1, 0], x)
        
    for x in range(n):
        qc.h(x)

    qc.append(q_uf, [x for x in range(n * 2)])

    # sanity check

    v = get_statevector(qc).to_dict()
    for k, _ in v.items():

        x, fx = k[n: 2 * n], k[:n]
        x, fx = int(x[::-1], 2), int(fx[::-1], 2)
        
        assert(fx == f[x])

    # --------

    for x in range(n, n * 2):
        qc.measure(x, x)

    #v = get_statevector(qc).to_dict()
    #print(v)

    for x in range(n):
        qc.h(x)
        qc.measure(x, x)

    qc.draw('mpl')

    for _ in range(n + 3):
        
        for res in simulate_runs(qc, runs = 1).keys():
            w = res[n: 2 * n]
            w = int(w[::-1], 2)
            break

        # solving the system manually
        # here just checking for s * w = 0

        dot_prod = s & w
        s_ = 0
        while dot_prod > 0:
            s_ ^= dot_prod & 0x01
            dot_prod >>= 1

        assert(s_ == 0)

    plot_simulated_runs(qc, runs=128)
    #print(get_statevector(qc))

    plt.show()

def grover_alg():
    '''
        n = 4\n
        generates random f: {0, 1}^n -> {0, 1} with exactly one x s.t. f(x) = 1
    '''

    n = 4
    
    def _init_uniform_superposition():
        
        # also adds ancilla qubit initialized to |->

        unif = QuantumCircuit(n + 1)
        unif.initialize([0, 1], n)

        unif.h([qb for qb in range(n + 1)])

        return unif

    def _get_random_Uf():
        
        x_1 = random.randint(0, (1 << n) - 1)
        f = {x: 0 for x in range(1 << n)}
        f[x_1] = 1

        print(f"f({x_1}) = 1")

        Uf = [[0 for _ in range(1 << (n + 1))] for _ in range(1 << (n + 1))]

        for k in range(1 << (n + 1)):
            Uf[k ^ f[k >> 1]][k] = 1

        Uf = qiskit_tensor_conversion(Uf, n + 1)
        q_uf = Operator(Uf)

        return q_uf

    def _get_U0():

        U0 = [[0 for _ in range(1 << n)] for _ in range(1 << n)]

        for k in range(1 << n):
            U0[k][k] = -1

        U0[0][0] = 1

        U0 = qiskit_tensor_conversion(U0, n)
        q_u0 = Operator(U0)

        return q_u0

    def _get_grover_iterate(q_uf, q_u0):

        g = QuantumCircuit(n + 1)

        g.append(q_uf, [qb for qb in range(n + 1)])
        
        g.h([qb for qb in range(n)])
        g.append(q_u0, [qb for qb in range(n)])
        g.h([qb for qb in range(n)])

        return g

    def _measure_output(qc: QuantumCircuit):
        
        qc.add_register(ClassicalRegister(n))
        for qb in range(n):
            qc.measure(qb, qb)

        plot_simulated_runs(qc)

    qc = _init_uniform_superposition()

    q_uf = _get_random_Uf()
    q_u0 = _get_U0()

    for _ in range(int(np.floor(np.sqrt(n)))):
        qc.append(_get_grover_iterate(q_uf, q_u0), [qb for qb in range(n + 1)])

    #qc.draw('mpl')
    _measure_output(qc)

    plt.show()

def start():

    #example()
    #superdense_coding_real(1, 1)
    #quantum_teleportation(np.sqrt(3) / 2, 0.5)
    #equiv_test()
    #deutsch_josza()
    #simon_alg()
    grover_alg()

    return

if __name__ == "__main__":
    start()