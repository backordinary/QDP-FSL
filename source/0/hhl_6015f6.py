# https://github.com/foxwy/QELM-VQELM/blob/3178820091cebf19d001937afac7827f98c545fe/basic_function/HHL.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import Operator
from qiskit import Aer
import numpy as np
from math import pi, sqrt
import math
from scipy.linalg import expm

np.set_printoptions(precision=5, linewidth=400)

############################
### function definitions ###
############################
def Print_param(params, params_name):
    print('-'*30)
    for i in range(len(params)):
        print(params_name[i]+':', params[i])
    print('-'*30)


def Print_title(title):
    print('*'*40)
    print('*'*(20-int(len(title)/2))+title+'*'*(20+int(len(title)/2)-len(title)))
    print('*'*40)


def printcircuitunitary(circ):
    bkend = Aer.get_backend('unitary_simulator')
    job = execute(circ, bkend)
    result = job.result()
    print(result.get_unitary(circ, decimals=3))


def printstatevector(circ):
    bkend = Aer.get_backend('statevector_simulator')
    job = execute(circ, bkend)
    result = job.result()
    print(result.get_statevector())


def getstatevector(circ):
    bkend = Aer.get_backend('statevector_simulator')
    job = execute(circ, bkend)
    result = job.result()
    return result.get_statevector()


def qft(circ, q, n):
    for j in range(n):
        for k in range(j):
            circ.cp(pi / float(2**(j - k)), q[j], q[k])
        circ.h(q[j])


def iqft(circ, q, n):
    for j in range(n - 1, -1, -1):
        circ.h(q[j])
        for k in range(j - 1, -1, -1):
            circ.cp(-pi / float(2**(j - k)), q[k], q[j])


def getUs(clocksize, mat):
    Ulist = list()
    invUlist = list()
    Ulist.append(Operator(mat))
    invUlist.append(Operator(mat.conj().T))
    for i in range(clocksize - 1):
        mat = np.matmul(mat, mat)
        Ulist.append(Operator(mat))
        invUlist.append(Operator(mat.conj().T))

    return Ulist, invUlist


def bintoint(string):
    number = 0
    for i in range(len(string)):
        number = number + int(string[len(string) - 1 - i]) * 2**i
    return number


def binfractodec(string):
    number = 0
    for i in range(len(string)):
        number = number + int(string[len(string) - 1 - i]) * 2**(-i - 1)
    return number


def hermtocontU(mat, T):
    # takes uncontrolled matrix
    # must be hermitian currently
    # - add code to make non-hermitian
    # matrix hermitian
    if np.array_equal(mat, mat.conj().T):
        hermop = mat
    expherm = expm(2j * pi * hermop / T)

    # add control
    M0 = np.asarray([[1, 0],
                     [0, 0]])
    M1 = np.asarray([[0, 0],
                     [0, 1]])
    I = np.eye(np.shape(hermop)[0])
    cexpherm = np.kron(M0, I) + np.kron(M1, expherm)

    return cexpherm


def prepareb(vector, circ, qb):
    # add circuit elements to prepare the state
    # the normalized vector as a state

    normfactor = np.linalg.norm(vector)
    state = vector / normfactor

    circ.initialize(state, [qb_i for qb_i in qb])
    circ.barrier()

    return normfactor


def swap_test(x1, x2, nqubit):
    qc = QuantumCircuit(2 * nqubit + 1)
    qc.h(2 * nqubit)
    qc.initialize(x1, list(range(nqubit)))
    qc.initialize(x2, list(range(nqubit, 2 * nqubit)))
    qc.barrier()
                  
    for i in range(nqubit):
        qc.cswap(2 * nqubit, i, i + nqubit)
    qc.barrier()
    
    qc.h(2 * nqubit)
    qc.save_statevector()
    # display(qc.draw())
    
    sim = Aer.get_backend('aer_simulator') 
    counts = sim.run(qc).result().get_counts()
    #plot_histogram(counts)
    
    total_counts = 0
    for key in counts.keys():
        if key[0] == '0':
            total_counts += counts[key]

    value = sqrt(total_counts * 2 - 1)
    
    return value


def Cal_C_swap(b, A, qbtoxstate):
    mean = 0
    idx = 0
    qb_norm = np.linalg.norm(qbtoxstate)
    for i in range(len(b)):
        A_i_norm = np.linalg.norm(A[i])
        b_pre = swap_test(A[i] / A_i_norm, qbtoxstate/qb_norm, int(math.log(len(qbtoxstate), 2))) * A_i_norm * qb_norm
        if b[i] != 0 and b_pre != 0:
            mean += abs(b[i]) / abs(b_pre)  # ????
            idx += 1
    return mean / idx


def Cal_C(b, A, qbtoxstate):
    mean = 0
    idx = 0
    b_pre = A.dot(qbtoxstate)
    for i in range(len(b)):
        if b[i] != 0 and b_pre[i] != 0:
            mean += b[i] / b_pre[i]
            idx += 1
    return mean / idx


def hhl(A, b, T, clocksize, r):
    actualans = np.matmul(np.linalg.inv(A), np.asarray(b).reshape(len(b), 1))

    w, v = np.linalg.eig(A)
    '''
    print('Eigenvalues, eigenvectors of A:\n', w / T, v)
    print('-'*30)'''

    cexpherm = hermtocontU(A, T)

    # qbtox size wont work if b and cepherm dimensions are not a power of 2
    # need to modify hermtocontU() and prepareb() to make sure they produce
    # stuff with dimension 2^n by 2^n. this may change how the result is to be
    # interpreted, so hermtocontU() should also return flags indicating
    # how to proceed in later parts of the algorithm

    qclock = QuantumRegister(clocksize, 'clk')
    qbtox = QuantumRegister(np.log2(np.shape(b)[0]), 'btox')
    qanc = QuantumRegister(1, 'anc')
    cclock = ClassicalRegister(clocksize, 'cclk')
    cbtox = ClassicalRegister(np.log2(np.shape(b)[0]), 'cbtox')
    canc = ClassicalRegister(1, 'canc')

    circ = QuantumCircuit(qbtox, qclock, qanc, cbtox, cclock, canc)

    bnormfactor = prepareb(b, circ, qbtox)

    ################################
    ### quantum phase estimation ###
    ################################

    # should quantum phase estimation be run
    # first and then eigenvalues measured
    # to get the condition number
    # not efficient but would be better for
    # understanding/explaining how things work

    circ.h(qclock)
    circ.barrier()

    Ulist, _ = getUs(len(qclock), cexpherm)
    for i in range(len(Ulist)):
        reglist = [qbtox_i for qbtox_i in qbtox]
        reglist.append(qclock[i])
        circ.unitary(Ulist[i], reglist)
    circ.barrier()

    iqft(circ, qclock, len(qclock))
    circ.barrier()

    #####################
    ### rotation part ###
    #####################

    # need to understand role of registers M and L
    # and implement them

    # fidelity of answer goes up with r
    # probability of ancilla = 1 for post selection goes down with r
    for i in range(len(qclock)):
        circ.mcry(1 / ((2**(-i)) * (2**r)), [qclock[i]], qanc[0], q_ancillae=None)
    circ.barrier()

    # for any two == 1
    for i in range(len(qclock)):
        for j in range(i + 1, len(qclock)):
            circ.mcry(-1 / ((2**(-i)) * (2**r)), [qclock[i], qclock[j]], qanc[0], q_ancillae=None)
            circ.mcry(-1 / ((2**(-j)) * (2**r)), [qclock[i], qclock[j]], qanc[0], q_ancillae=None)
            circ.mcry(1 / ((2**(-i) + 2**(-j)) * (2**r)), [qclock[i], qclock[j]], qanc[0], q_ancillae=None)
    circ.barrier()

    ####################
    ### Reverse QPE  ###
    ####################

    qft(circ, qclock, len(qclock))
    circ.barrier()

    _, invUlist = getUs(len(qclock), cexpherm)
    for i in range(len(invUlist)):
        reglist = [qbtox[k] for k in range(len(qbtox))]
        reglist.append(qclock[len(invUlist) - 1 - i])
        circ.unitary(invUlist[len(invUlist) - 1 - i], reglist)
    circ.barrier()

    circ.h(qclock)
    circ.barrier()

    #########################################################
    ### get statevector for qbtox conditioned on qanc = 1 ###
    #########################################################

    '''
    print('\n############################')
    print('### Statevector analysis ###')
    print('############################\n')'''

    statevec = getstatevector(circ)
    statevec = statevec.reshape(len(statevec), 1)
    binlen = (len(qclock) + len(qbtox) + len(qanc))
    zeros = '0' * binlen

    postselectionprob = 0
    postselectedvector = list()
    postselectedbinaryidx = list()

    # print('Full Statevector:\n', statevec)
    for i in range(len(statevec)):
        binary = str(bin(i))[2:]
        if len(binary) < binlen:
            binary = zeros[:-len(binary)] + binary
        # print(binary, statevec[i][0])
        if binary[0] == '1':
            postselectionprob = postselectionprob +\
                np.sqrt(np.real(statevec[i][0])**2 +
                        np.imag(statevec[i][0])**2)
            postselectedvector.append(statevec[i][0])
            postselectedbinaryidx.append(binary)
    postselectedvector = postselectedvector / postselectionprob

    #print('Postselected Statevector (Postselection prob - {:.2f}%)'.format(postselectionprob * 100))
    qbtoxstate = list()
    qbtoxbinidx = list()
    for i in range(len(postselectedvector)):
        # print(postselectedbinaryidx[i][1:],postselectedvector[i])
        if postselectedbinaryidx[i][1:1 + len(qclock)] == '0' * len(qclock):
            qbtoxbinidx.append(postselectedbinaryidx[i][-len(qbtox):])
            qbtoxstate.append(postselectedvector[i])

    '''
    print('Solution Statevector:')
    for i in range(len(qbtoxstate)):
        print(qbtoxbinidx[i], qbtoxstate[i])
    print('-'*40)'''

    #C = np.mean(actualans / np.real(np.asarray(qbtoxstate).reshape(len(qbtoxstate), 1)))
    #print('b, b*:\n', np.asarray(b).reshape(len(b), 1), A.dot(np.real(np.asarray(qbtoxstate).reshape(len(qbtoxstate), 1))))
    C_y = Cal_C(b, A, np.array(qbtoxstate).real)
    '''
    print('C_real:\n', C_y)
    print('-'*40)
    print('actual:\n', actualans, '\nstatevector:\n', np.asarray(qbtoxstate).reshape(len(qbtoxstate), 1))
    print('statevector times C:\n', np.asarray(qbtoxstate).reshape(len(qbtoxstate), 1) * C_y)
    print('-'*40)'''

    return actualans, (np.asarray(qbtoxstate).reshape(len(qbtoxstate), 1) * C_y).real

    #####################################
    ### measure, analyze measurements ###
    #####################################

    '''
    circ.measure(qanc, canc)
    circ.measure(qbtox, cbtox)
    circ.measure(qclock, cclock)

    print('\n###############')
    print('### Circuit ###')
    print('###############\n')
    print(circ)

    print('\n############################')
    print('### Measurement analysis ###')
    print('############################\n')

    shots = 100000
    bkend = Aer.get_backend('qasm_simulator')
    job = execute(circ, bkend, shots=shots)
    result = job.result()
    counts = result.get_counts(circ)
    print(counts)


    print('Need to fix this part')

    countpercent = np.zeros(shape=(len(qanc) + len(qbtox) + len(qclock), 1))
    totcounts = 0
    for key in counts.keys():
        keysp = key.split(' ')
        if keysp[0] == '1':
            totcounts = totcounts + counts[key]
            countpercent[0] = countpercent[0] + counts[key]
            for i in range(len(keysp[1])):
                if keysp[1][i] == '1':
                    countpercent[1 + i] = countpercent[1 + i] + counts[key]
            for i in range(len(keysp[2])):
                if keysp[2][i] == '1':
                    countpercent[1 + len(keysp[1]) + i] = countpercent[1 + len(keysp[1]) + i] + counts[key]

    countpercent = 100 * countpercent / totcounts
    print('-----------------------------------------------------------')
    print('probability of ancilla = 1 for post-selection from measurement: ', 100 * totcounts / shots, '%')
    # print('-----------------------------------------------------------')
    #print('percent probabilities of qubits = 1, conditioned on ancilla = 1:\n', countpercent)

    # get probabilities
    totcounts = 0
    HHLans = np.zeros(shape=(2**len(qbtox), 1))
    for key in counts.keys():
        keysp = key.split(' ')
        if keysp[0] == '1':
            totcounts = totcounts + counts[key]
            for i in range(len(qbtox)):
                HHLans[bintoint(keysp[2])] = counts[key]
    HHLans = HHLans / totcounts

    actualans = np.matmul(np.linalg.inv(A), np.asarray(b).reshape(len(b), 1))

    print('State probabilities from HHL:')
    for i in range(np.shape(HHLans)[0]):
        print('{}|{}>'.format(HHLans[i][0], i), end=' ')
        if i < np.shape(HHLans)[0] - 1:
            print('+', end=' ')


    print('\n-----------------------------------------------------------')
    print('Square root of these probabilities is proportional to (the magnitudes of) the exact solution, x:')
    print('|x,HHL> =', np.sqrt(HHLans.T)[0])
    print('-----------------------------------------------------------')
    print('The actual solution is:')
    print('x =', actualans.T[0], 'and the absolute value of x\'s elements should equal C|x,HHL>')
    print('-----------------------------------------------------------')
    print('|x|/|x,HHL> element-wise is')
    print(np.abs(actualans)/np.sqrt(HHLans))
    print('-----------------------------------------------------------')
    print('We can take the average of these elements as C')
    print('C =', np.mean(np.abs(actualans)/np.sqrt(HHLans)))
    print('-----------------------------------------------------------')
    print('Now we can show that |x,HHL> is proportional to the magnitudes of elements of x')
    print('C|x,HHL> =', np.sqrt(HHLans.T)[0]*np.mean(np.abs(actualans)/np.sqrt(HHLans)), 'is approximately equal to absolute value of x\'s elements,', np.abs(actualans.T[0]))
    print('-----------------------------------------------------------')
    '''


def defaultprograms(num):
    if num == '1':
        # LANL Example
        A = np.asarray([[0.75, 0.25],
                        [0.25, 0.75]])
        b = np.asarray([2, 0])
        T = 2
        clocksize = 2
        r = 4

    if num == '2':
        # From the paper, 'Quantum Circuit Design for Solving
        # Linear Systems of Equations'
        A = 0.25 * np.asarray([[15, 9, 5, -3],
                               [9, 15, 3, -5],
                               [5, 3, 15, -9],
                               [-3, -5, -9, 15]])
        b = 0.5 * np.asarray([1, 1, 1, 1])
        T = 16
        clocksize = 4
        r = 5

    if num == '3':
        # Example with matrix that doesn't have eigenvalues
        # that are a power of 0.5 but that are an exact
        # sum of low powers of 0.5
        A = 2 * np.asarray([[0.375, 0],
                            [0, 0.25]])
        b = np.asarray([1, 1])
        T = 2
        clocksize = 4
        r = 4

    return A, b, T, clocksize, r


def To_Hermitian(A, b):
    if np.array_equal(A, A.conj().T):  # hermitian matrix
        #print('A is Hermitian matrix!')
        return A, b, 1
    else:
        '''
        print('A is not a Hermitian matrix!')
        print('Fill A to be a Hermitian matrix...')'''
        A_n = len(A)
        H = np.block([[np.zeros((A_n, A_n)), A], \
                      [A.conj().T, np.zeros((A_n, A_n))]])  # [[0, A], [A+, 0]]
        T = np.block([b, np.zeros(A_n)])  # [b, 0]
        return H, T, 0


# calculate T, clocksize, r according to A
def Less_value(lis, value):  # judge all elements of list less than value
    a = np.array(lis)
    return (np.abs(a) < value).all()


def Cal_param(A):
    # eigenvalues of A/T are all less than 1
    w, _ = np.linalg.eig(A)
    T = 1
    flag = 1
    while flag:
        Less_one = Less_value(w/T, 1)
        if Less_one:
            flag = 0
        else:
            T *= 2
    T *= 2

    # clocksize
    clocksize = len(A)

    # r
    r = len(A)

    return T, clocksize, r


def Cal_A_b_HHL(H, T):
    #rint_title('A, b, T, clocksize, r')
    #-----A, b-----
    #Print_param([H, T], ['A', 'b'])

    #-----parameter-----
    A, b, herm_flag = To_Hermitian(H, T)
    T, clocksize, r = Cal_param(A)
    #Print_param([A, b, T, clocksize, r], ['A', 'b', 'T', 'clocksize', 'r'])

    #-----HHL-----
    #Print_title('HHL')
    actual_vector, pre_vector = hhl(A, b, T, clocksize, r)
    if herm_flag == 0:  # non-hermitian matrix
        actual_vector = actual_vector[int(len(actual_vector)/2):]
        pre_vector = pre_vector[int(len(pre_vector)/2):]
    #print('actual_vector, pre_vector:\n', actual_vector, pre_vector)

    return actual_vector, pre_vector


if __name__ == '__main__':
    #-----A, b-----
    # A, b, T, clocksize, r = defaultprograms(1)
    H = np.asarray([[1, 1], [2, 0]])
    T = np.asarray([2, 2])
    Cal_A_b_HHL(H, T)
