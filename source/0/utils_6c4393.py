# https://github.com/MKrbm/QBM/blob/359c3a570c146920789bb22e4b4d343020bd68eb/functions/utils_.py
import numpy as np
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import *
from qiskit.circuit import Parameter, parameter
from qiskit.quantum_info.operators import Operator
import copy

class estimate_params():

    def __init__(self, Vs, Vs_qargs, gs, gs_qargs, ham_dict , n, params=None, backend_name = None):

        """

        ex)

        backend_name = FakeSydney

        """

        
        self.Vs = Vs
        self.Vs_qargs = Vs_qargs
        self.gs = gs
        self.gs_qargs = gs_qargs
        self.ham_dict = ham_dict
        self.n = n
        self.params = params 



        if backend_name is None:

            self.backend = Aer.get_backend('aer_simulator')


        else:

            assert type(backend_name) is str, 'backend_name must be string'

            try:
                device_backend = eval(backend_name+"()")
                self.backend = AerSimulator.from_backend(device_backend)
            except:
                print("backend_name didn't match any backend")
                print("Therefore, aer_simulator is choosen")
                self.backend = Aer.get_backend('aer_simulator')


    def estimate_Fisher(self, der_pos = [0,1], NUM=10000):

        assert self.params, 'please provide params'

        N = len(self.Vs)
        L = 1


        q = QuantumRegister(self.n+1)
        c = ClassicalRegister(1)
        A = QuantumCircuit(q,c, name='Fisher')
        A.h(0)


        for i in range(N):
            if i in der_pos:
                A.x(0)
                A.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
            A.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
        A.h(0)
        A.measure(0, 0)

        A = A.bind_parameters(self.params)


        if NUM > 10000:
            L = NUM//10000
            NUM = 10000
        
        A_results = []

        for _ in range(L):
            
            # backend = BasicAer.get_backend('qasm_simulator')
            result = execute(A, self.backend, shots=NUM).result()
            counts  = result.get_counts(A)
            if '0' in counts.keys():
                A_results.append((counts['0']/NUM)*2 -1)
            else:
                A_results.append(-1)

        return np.mean(A_results)


    def estimate_C(self ,der_pos = [1] ,NUM = 10000):

        N = len(self.Vs)
        M = len(self.ham_dict['ham'])
        L = 1


        C_hat = 0

        q = QuantumRegister(self.n+1)
        c = ClassicalRegister(1)



        for m in range(M):


            C = QuantumCircuit(q,c, name='Fisher')
            C.h(0)
            C.rz(np.pi/2,0)
            


            for i in range(N):
                if i in der_pos:
                    C.x(0)
                    C.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
                C.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
            C.x(0)
            C.append(
                self.ham_dict['ham'][m],
                [q[k] for k in self.ham_dict['pos'][m]]
                )
            C.h(0)
            C.measure(0, 0)

            C = C.bind_parameters(self.params)



            # backend = BasicAer.get_backend('qasm_simulator')

            if NUM > 10000:
                L = NUM//10000
                NUM = 10000
            
            C_results = []

            for _ in range(L):
                
                # backend = BasicAer.get_backend('qasm_simulator')
                result = execute(C, self.backend, shots=NUM).result()
                counts  = result.get_counts(C)
                if '0' in counts.keys():
                    C_results.append((counts['0']/NUM)*2 -1)
                else:
                    C_results.append(-1)
                
            
            C_hat += np.mean(C_results) * (-1) * self.ham_dict['c'][m] 


        return C_hat


    def estimate_AC(self, NUM = 10000):

        N = len(self.Vs)

        C_array = np.zeros(N)
        A_array = np.eye(N)

        for i in range(N):
            for j in range(i):
                A_array[i,j] = \
                    self.estimate_Fisher(der_pos = [i, j] ,NUM = NUM)
                A_array[j, i] = A_array[i,j]

        for i in range(N):
            C_array[i] = \
                self.estimate_C(der_pos = [i] ,NUM = NUM)
        
        return A_array/4, C_array/2

class estimate_params2():

    def __init__(self,VQS, params=None):

        self.VQS = VQS
        self.Vs = VQS.V
        self.Vs_qargs = VQS.pos_est
        self.gs = VQS.gs
        self.gs_qargs = VQS.g_pos
        self.ham_dict = VQS.o_ham_dict
        self.n = VQS.n
        self.params = params 

    def estimate_A(self, der_pos = 0, NUM=10000):

        N = len(self.Vs)
        M = len(self.ham_dict['ham'])
        L = 1



        q = QuantumRegister(self.n+1)


        C = QuantumCircuit(q, name='Fisher')
        C.h(0)
        C.rz(np.pi/2,0)
        


        for i in range(N):
            if i == der_pos:
                C.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
            C.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
        C.h(0)
        C.measure_all()

        C = C.bind_parameters(self.params)
        
        
        # backend = BasicAer.get_backend('qasm_simulator')
        result = execute(C, self.backend, shots=NUM).result()
        result  = result.get_counts(C)

        states = []
        counts = []


        val = np.zeros((len(self.ham_dict['ham']), len(result)))


        for n, (key, item) in enumerate(result.items()):

            state_array = np.zeros(len(key), np.int8)
            for i,j in enumerate(key):
                state_array[-(i+1)] = -((int(j) * 2) - 1)
            states.append(state_array)
            counts.append(item)
            for m, pos in enumerate(self.ham_dict['pos']):
                
                pos_ = [0] + [i+1 for i in pos]
                val[m,n] = state_array[pos_].prod()
                
        counts = np.array(counts)

        
        
        return (val * counts/counts.sum()).sum(axis=-1)


        # if NUM > 10000:
        #     L = NUM//10000
        #     NUM = 10000
        
        # C_results = []

        # for _ in range(L):
            
        #     backend = BasicAer.get_backend('qasm_simulator')
        #     result = execute(C, backend, shots=NUM).result()
        #     counts  = result.get_counts(C)
        #     if '0' in counts.keys():
        #         C_results.append((counts['0']/NUM)*2 -1)
        #     else:
        #         C_results.append(-1)
            
        
        # C_hat += np.mean(C_results) * (-1) * self.ham_dict['c'][m] 


        return C_hat


    def estimate_C(self ,NUM = 10000):

        VQS = self.VQS

        assert self.params is not None

        # backend = BasicAer.get_backend('qasm_simulator')

        state = VQS.state().bind_parameters(self.params)
        state.measure_all()
        result = execute(state, self.backend, shots=NUM).result()
        result  = result.get_counts(state)

        states = []
        counts = []
        val = np.zeros((len(self.ham_dict['ham']), len(result)))


        for n, (key, item) in enumerate(result.items()):
            
            state_array = np.zeros(len(key), np.int8)
            for i,j in enumerate(key):
                state_array[-(i+1)] = -((int(j) * 2) - 1)
            states.append(state_array)
            counts.append(item)
            for m, pos in enumerate(self.ham_dict['pos']):
                val[m,n] = state_array[pos].prod()

        counts = np.array(counts)
        ham = np.array(self.ham_dict['c']) @ val 
        C = (val * ham * counts/counts.sum()).sum(axis=-1)

        return C


    def estimate_AC(self, NUM = 10000):

        N = len(self.ham_dict['ham'])
        M = len(self.Vs)


        C_array = np.zeros(N)
        A_array = np.zeros((N, M))

        for m in range(M):
            A_array[:,m] = \
                self.estimate_A(der_pos = m ,NUM = NUM)

        C_array = \
            self.estimate_C(NUM = NUM)
        
        return A_array/2, C_array



class estimate_params3():

    def __init__(self,VQS, backend_name):

        """

        ex)
        backend_name = FakeSydney

        
        """

        self.VQS = VQS
        self.Vs = VQS.V
        self.Vs_qargs = VQS.pos_est
        self.gs = VQS.gs
        self.gs_qargs = VQS.g_pos
        self.ham_dict = VQS.o_ham_dict
        self.n = int(VQS.n)
        # self.set_theta(theta)

        if backend_name is None:

            self.backend = Aer.get_backend('aer_simulator')


        else:

            assert type(backend_name) is str, 'backend_name must be string'

            try:
                device_backend = eval(backend_name+"()")
                self.backend = AerSimulator.from_backend(device_backend)

                # print("backend : ", self.backend)
            except:
                print("backend_name didn't match any backend")
                print("Therefore, aer_simulator is choosen")
                self.backend = Aer.get_backend('aer_simulator')



    def estimate_A(self, NUM=10000):




        A_array = np.zeros((self.VQS.n_params,self.VQS.n_params))
        for i_c in range(0+1):


            if 0 == i_c:
                q = QuantumRegister(self.VQS.n)
                A = QuantumCircuit(q, name='Fisher')


                for i in range(0 * self.VQS.n):
                    A.append(self.VQS.V[i],[q[k] for k in self.VQS.pos[i]])
                A.rx(np.pi/2, q)
                A.measure_all()
                params = {self.VQS.P[i] : self.theta[i] for i in range(0 * self.VQS.n)}

                A = A.bind_parameters(params)
                # backend = BasicAer.get_backend('qasm_simulator')
                result = execute(A, self.backend, shots=NUM).result()
                result  = result.get_counts(A)
                
                states = []
                counts = []
                val = np.zeros((len(result), self.VQS.n, self.VQS.n))

                for n, (key, item) in enumerate(result.items()):

                    state_array = np.zeros(len(key), np.int8)
                    for i,j in enumerate(key):
                        state_array[-(i+1)] = -((int(j) * 2) - 1)
                        local_A = state_array[:,None] @ state_array[None,:]
                    val[n,:,:] = local_A
                    states.append(state_array)
                    counts.append(item)
                    # print(state_array)


                counts = np.array(counts)
                # print(result)

                val_ = (val * counts[:,None,None]/counts.sum()).sum(axis=0)


                for i in range(self.VQS.n):
                    i_ = i + i_c * self.VQS.n
                    for j in range(self.VQS.n):
                        j_ = j + 0 * self.VQS.n
                        # print(i_,j_, val_[i, j])
                        A_array[i_, j_] = val_[i,j]

            elif 0 != i_c:
                
                for i_r in range(self.VQS.n):
                    q = QuantumRegister(self.VQS.n+1)
                    A = QuantumCircuit(q, name='Fisher')
                    A.h(0)

                    for i in range(0 * self.VQS.n):
                        if i == i_c*self.VQS.n:
                            A.barrier()
                            A.append(self.VQS.gs[i_r+i_c*self.VQS.n], [q[k] for k in self.VQS.g_pos[i_r+i_c*self.VQS.n]])
                            A.barrier()
                        A.append(self.VQS.V[i],[q[k] for k in self.VQS.pos_est[i]])

                    A.h(0)
                    A.rx(np.pi/2, q[1:])
                    A.measure_all()
                    params = {self.VQS.P[i] : self.theta[i] for i in range(0 * self.VQS.n)}

                    A = A.bind_parameters(params)
                    # backend = BasicAer.get_backend('qasm_simulator')
                    result = execute(A, self.backend, shots=NUM).result()
                    result  = result.get_counts(A)

                    states = []
                    counts = []
                    val = np.zeros((len(result), self.VQS.n))


                    for n, (key, item) in enumerate(result.items()):

                        state_array = np.zeros(len(key), np.int8)
                        for i,j in enumerate(key):
                            state_array[-(i+1)] = -((int(j) * 2) - 1)
                        states.append(state_array[1:]*state_array[0])
                        counts.append(item)
                        val[n] = state_array[1:]*state_array[0]

                    counts = np.array(counts)
                    val_ = ((val * counts[:,None])/counts.sum()).sum(axis=0)

                    for j in range(self.VQS.n):
                        j_ = j + 0 * self.VQS.n

                        A_array[i_c * self.VQS.n + i_r, j_] = val_[j]
                        A_array[j_, i_c * self.VQS.n + i_r] = val_[j]
        for i_c in range(1+1):


            if 1 == i_c:
                q = QuantumRegister(self.VQS.n)
                A = QuantumCircuit(q, name='Fisher')


                for i in range(1 * self.VQS.n):
                    A.append(self.VQS.V[i],[q[k] for k in self.VQS.pos[i]])
                A.rx(np.pi/2, q)
                A.measure_all()
                params = {self.VQS.P[i] : self.theta[i] for i in range(1 * self.VQS.n)}

                A = A.bind_parameters(params)
                # backend = BasicAer.get_backend('qasm_simulator')
                result = execute(A, self.backend, shots=NUM).result()
                result  = result.get_counts(A)
                
                states = []
                counts = []
                val = np.zeros((len(result), self.VQS.n, self.VQS.n))

                for n, (key, item) in enumerate(result.items()):

                    state_array = np.zeros(len(key), np.int8)
                    for i,j in enumerate(key):
                        state_array[-(i+1)] = -((int(j) * 2) - 1)
                        local_A = state_array[:,None] @ state_array[None,:]
                    val[n,:,:] = local_A
                    states.append(state_array)
                    counts.append(item)
                    # print(state_array)


                counts = np.array(counts)
                # print(result)

                val_ = (val * counts[:,None,None]/counts.sum()).sum(axis=0)


                for i in range(self.VQS.n):
                    i_ = i + i_c * self.VQS.n
                    for j in range(self.VQS.n):
                        j_ = j + 1 * self.VQS.n
                        # print(i_,j_, val_[i, j])
                        A_array[i_, j_] = val_[i,j]

            elif 1 != i_c:
                
                for i_r in range(self.VQS.n):
                    q = QuantumRegister(self.VQS.n+1)
                    A = QuantumCircuit(q, name='Fisher')
                    A.h(0)

                    for i in range(1 * self.VQS.n):
                        if i == i_c*self.VQS.n:
                            A.barrier()
                            A.append(self.VQS.gs[i_r+i_c*self.VQS.n], [q[k] for k in self.VQS.g_pos[i_r+i_c*self.VQS.n]])
                            A.barrier()
                        A.append(self.VQS.V[i],[q[k] for k in self.VQS.pos_est[i]])

                    A.h(0)
                    A.rx(np.pi/2, q[1:])
                    A.measure_all()
                    params = {self.VQS.P[i] : self.theta[i] for i in range(1 * self.VQS.n)}

                    A = A.bind_parameters(params)
                    # backend = BasicAer.get_backend('qasm_simulator')
                    result = execute(A, self.backend, shots=NUM).result()
                    result  = result.get_counts(A)

                    states = []
                    counts = []
                    val = np.zeros((len(result), self.VQS.n))


                    for n, (key, item) in enumerate(result.items()):

                        state_array = np.zeros(len(key), np.int8)
                        for i,j in enumerate(key):
                            state_array[-(i+1)] = -((int(j) * 2) - 1)
                        states.append(state_array[1:]*state_array[0])
                        counts.append(item)
                        val[n] = state_array[1:]*state_array[0]

                    counts = np.array(counts)
                    val_ = ((val * counts[:,None])/counts.sum()).sum(axis=0)

                    for j in range(self.VQS.n):
                        j_ = j + 1 * self.VQS.n

                        A_array[i_c * self.VQS.n + i_r, j_] = val_[j]
                        A_array[j_, i_c * self.VQS.n + i_r] = val_[j]

        return A_array/4


    def estimate_C(self, der_pos = 0 ,NUM = 10000):

        N = len(self.Vs)
        M = len(self.ham_dict['ham'])
        L = 1



        q = QuantumRegister(self.n+1)


        C = QuantumCircuit(q, name='Fisher')
        C.h(0)
        C.rz(np.pi/2,0)
        


        for i in range(N):
            if i == der_pos:
                C.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
            C.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
        C.h(0)
        C.measure_all()

        C = C.bind_parameters(self.params)
        
        
        # backend = BasicAer.get_backend('qasm_simulator')
        result = execute(C, self.backend, shots=NUM).result()
        result  = result.get_counts(C)

        states = []
        counts = []


        val = np.zeros((len(self.ham_dict['ham']), len(result)))


        for n, (key, item) in enumerate(result.items()):

            state_array = np.zeros(len(key), np.int8)
            for i,j in enumerate(key):
                state_array[-(i+1)] = -((int(j) * 2) - 1)
            states.append(state_array)
            counts.append(item)
            for m, pos in enumerate(self.ham_dict['pos']):
                
                pos_ = [0] + [i+1 for i in pos]
                val[m,n] = state_array[pos_].prod()
                
        counts = np.array(counts)

        
        
        ham = np.array(self.ham_dict['c']) @ val 
        
        return (ham * counts/counts.sum()).sum(axis=-1)/2


    def estimate_AC(self, NUM = 10000):

        M = len(self.Vs)


        C_array = np.zeros(M)
        A_array = np.zeros((M, M))

        for m in range(M):

            C_array[m] = \
                self.estimate_C(NUM = NUM, der_pos=m)
        
        A_array = self.estimate_A(NUM)
        
        return A_array, C_array

    def set_theta(self, theta):

        self.theta = theta
        self.params = {self.VQS.P[i] : theta[i] for i in range(self.VQS.n_params)}





class VQS():

    def __init__(self, n, boltzmann=True):

        self.q = QuantumRegister(int(n))
        self.n_params = int(2*n)
        self.n = int(n)
        self.boltzmann = boltzmann

        self.P = [Parameter('θ{}'.format(i)) for i in range(self.n_params)]

        self.V = []
        self.pos = []

        for i in range(self.n):
            
            if (i+1)%self.n == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i))
                v.ry(self.P[i], self.n-1)
                for j in range(self.n):
                    v.cnot((self.n-1+j)%self.n, j%self.n)
                self.V.append(v)
                self.pos.append([i for i in range(self.n)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i))
                v.ry(self.P[i], 0)
                self.V.append(v)
                self.pos.append([i])
            

        for i in range(self.n):
            
            i_ = i + self.n

            if (i+1)%self.n == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i_))
                v.ry(self.P[i_], self.n-1)
                for j in range(int(self.n/2)):
                    v.cnot(j, j + int(self.n/2))

                if boltzmann:
                    v.ry(-np.pi/2,[i for i in range(self.n)])

                self.V.append(v)
                self.pos.append([i for i in range(self.n)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i_))
                if i < int(self.n/2):
                    v.ry(self.P[i_] + np.pi/2, 0)
                else:
                    v.ry(self.P[i_], 0)
                self.V.append(v)
                self.pos.append([i])


        # for i in range(self.n):

        #     v = QuantumCircuit(1, name='V{}'.format(i+2*self.n))
        #     v.ry(self.P[i+2*self.n], 0)
        #     self.V.append(v)
        #     self.pos.append([i])

        


        self.gs = []
        self.g_pos = []


        # q[self.n] is ancilla bit for computing A and C

        for i in range(self.n_params):
            g = QuantumCircuit(2, name='g{}'.format(i))
            g.cy(0,1)
            self.gs.append(g)
            self.g_pos.append([0, i%self.n+1])

        self.pos_est = [[j+1 for j in pos] for pos in self.pos]


    def set_ham(self, ham_dict):

        '''

        ham_dict['ham'] is list of {'zz', 'z'}

        ham_dict['pos'] is list of {[i,j] , [i]} which represent sites on which local hamiltonian act

        '''
        assert set(ham_dict.keys()) == set(['c','ham','pos']), 'contains wrong keys / missing some keys'


        for key, ele in ham_dict.items():
            assert len(ele) == len(ham_dict['ham']), "length of each elements doesn't match"

        self.o_ham_dict = copy.copy(ham_dict)

        self.ham_dict = {
            'ham' : [],
            'pos' : [],
            'c' : []
        }


    
        i = 0
        for h, pos, c in zip(ham_dict['ham'], ham_dict['pos'], ham_dict['c']):

            n = len(h)

            ham_list = ['x','y','z']
            # h_ = set(h)
            for h_ in set(h):
                assert h_ in ham_list,'invalid character in "ham" '
            

            lh = QuantumCircuit(n+1, name='ham{}'.format(i))

            for i, h_ in enumerate(h):
                
                if h_ == 'x':
                    lh.cx(0,i+1)
                elif h_ == 'y':
                    lh.cy(0,i+1)
                elif h_ == 'z':
                    
                    if self.boltzmann:
                        # lh.cx(0,i+1)
                        lh.cz(0,i+1)
                    else:
                        lh.cz(0,i+1)
            
            self.ham_dict['ham'].append(lh)
            self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            self.ham_dict['c'].append(c)

        

    def return_ham(self, return_list=False, boltzmann = None):

        if boltzmann is not None:
            B = boltzmann
        else:
            B = self.boltzmann

        
        N = int(self.n/2)
        i = 0
        ham = np.zeros((2**N,2**N), dtype=np.complex128)
        ham_list_ = []


        for h, pos, c in zip(self.o_ham_dict['ham'], self.o_ham_dict['pos'], self.o_ham_dict['c']):

            qc = QuantumCircuit(N)

            n = len(h)

            ham_list = ['x','y','z']
            # h_ = set(h)
            for h_ in set(h):
                assert h_ in ham_list,'invalid character in "ham" '
            


            for i, h_ in enumerate(h):
                
                if h_ == 'x':
                    qc.x(pos[i])
                elif h_ == 'y':
                    qc.y(pos[i])
                elif h_ == 'z':

                    if B:
                        qc.z(pos[i])
                    else:
                        qc.x(pos[i])

            
            ham += c*Operator(qc).data
            ham_list_.append(Operator(qc).data)

        if return_list:
            return ham, ham_list_, 


        return ham


    def state(self, boltzmann = None):


        rho = QuantumCircuit(self.q, name='rho')

        for i in range(self.n_params):
            rho.append(self.V[i], [self.q[j] for j in self.pos[i]])
        


        # assert n%2 == 0, 'n must be even number'

        return rho

        # n_params = n * 2


class VQS2(VQS):

    def __init__(self, n):

        self.q = QuantumRegister(int(n))
        self.n_params = int(n)
        self.n = int(n)

        self.P = [Parameter('θ{}'.format(i)) for i in range(self.n_params)]

        self.V = []
        self.pos = []
        self.n_prime = int(self.n/2)

        for i in range(self.n_prime):
            
            if (i+1)%self.n_prime == 0:
                v = QuantumCircuit(self.n_prime, name='V{}'.format(i))
                v.ry(self.P[i], self.n_prime-1)
                for j in range(self.n_prime):
                    v.cnot((self.n_prime-1+j)%self.n_prime, j%self.n_prime)
                self.V.append(v)
                self.pos.append([i for i in range(self.n_prime)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i))
                v.ry(self.P[i], 0)
                self.V.append(v)
                self.pos.append([i])
            
        for i in range(self.n_prime):
            
            i_ = i + self.n_prime

            if (i+1)%self.n_prime == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i_))
                v.ry(self.P[i_] + np.pi/2, self.n_prime-1)
                for j in range(int(self.n/2)):
                    v.cnot(j, j + int(self.n/2))
                self.V.append(v)
                self.pos.append([i for i in range(self.n)])


            else:
                v = QuantumCircuit(1, name='V{}'.format(i_))
                if i < int(self.n_prime/2):
                    v.ry(self.P[i_] + np.pi/2, 0)
                else:
                    v.ry(self.P[i_], 0)
                self.V.append(v)
                self.pos.append([i])



        self.gs = []
        self.g_pos = []


        # q[self.n] is ancilla bit for computing A and C

        for i in range(self.n_params):
            g = QuantumCircuit(2, name='g{}'.format(i))
            g.cy(0,1)
            self.gs.append(g)
            self.g_pos.append([0, i%self.n_prime+1])

        self.pos_est = [[j+1 for j in pos] for pos in self.pos]






