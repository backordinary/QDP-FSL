# https://github.com/leductruyen/VQE_Cluster/blob/7db6342edaf1b7c6100d46774b725a4479968335/TRUYEN_VQE_Running_Cluster/VQE_qubit_running_cluster.py
from ast import main
from qiskit.circuit.library import TwoLocal, EfficientSU2, RealAmplitudes, NLocal, PauliTwoDesign
# from qiskit_nature.circuit.library import UCCSD, PUCCD, SUCCD # chemistry simulation ansatzes, for example the UCCSD ansatz prepares a state where tuning the parameters turns excitations on and off
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, SLSQP
import numpy as np 
from numpy import real, sqrt, pi, linalg
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.result import marginal_counts
from IPython.display import display, clear_output
from qiskit import Aer, transpile, assemble
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector, plot_bloch_vector, plot_state_qsphere
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp, PauliOp
from qiskit_optimization import QuadraticProgram
from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ, execute
from dask.distributed import LocalCluster, Client
def main_function():
    def Expectation_Exact(gate=None,q1=None,q2=None,circ=None,realbackend=Aer.get_backend('qasm_simulator')):
        qc = QuantumCircuit(circ.num_qubits,circ.num_qubits)
        qc.append(circ,[i for i in range(circ.num_qubits)])
        
        # backend = Aer.get_backend('qasm_simulator')
        # noise_model = NoiseModel.from_backend(realbackend)
        backend = realbackend
        shot=20000
        
        # Get string of state
        qubit = qc.num_qubits
        bitq1 = '' # Get string of |...01...> state
        bitq2 = '' # Get string of |...10...> state
        bit0 = ''  # Get string of |...00...> state
        for i in range(qubit):
            bit0 += '0'
            if i == q1:
                bitq1 += '1'
                bitq2 += '0'
            elif i == q2:
                bitq2 += '1'
                bitq1 += '0'
            else:
                bitq1 += '0'
                bitq2 += '0'
        bit0 = bit0[::-1]
        bitq1 = bitq1[::-1]
        bitq2 = bitq2[::-1]
        
        if gate == "ZZ":
            qc.measure([q1,q2],[q1,q2])
            tqc = transpile(qc, backend,optimization_level=3)
            qobj = assemble(tqc,shots=shot)
            result = backend.run(qobj).result().get_counts(qc)
            for output in [bit0,bitq1,bitq2]:
                if output in result:
                    result[output]/=shot
                else:
                    result[output] = 0
            firesult=4*(result[bit0])-1-(2*(result[bit0]+result[bitq1])-1)-(2*(result[bit0]+result[bitq2])-1)# =(4P(i,j=0)-1-(2P(i=0)-1)-(2P(j=0)-1))
        elif gate == "Z":
            qc.measure(q1,q1)  
            tqc = transpile(qc, backend,optimization_level=3)
            qobj = assemble(tqc,shots=shot)
            result = backend.run(qobj).result().get_counts(qc)
            if bit0 in result:
                result[bit0]/=shot
            else:
                result[bit0] = 0
            firesult=(2*(result[bit0])-1)# =(2P(i=0)-1)
        elif gate == "X":
            qc.h(q1) # Apply H gate to apply X basic measurement
            qc.measure(q1,q1)  
            tqc = transpile(qc, backend,optimization_level=3)
            qobj = assemble(tqc,shots=shot)
            result = backend.run(qobj).result().get_counts(qc)
            if bit0 in result:
                result[bit0]/=shot
            else:
                result[bit0] = 0
            firesult=(2*(result[bit0])-1)# =(2P(i=0)-1)
        elif gate == "YY":
            qc.sdg(q1) # Apply Sdg gate to transform from X basic to Y basic measurement
            qc.h(q1) # Apply H gate to apply X basic measurement
            qc.sdg(q2) # Apply Sdg gate to transform from X basic to Y basic measurement
            qc.h(q2) # Apply H gate to apply X basic measurement
            qc.measure([q1,q2],[q1,q2])
            tqc = transpile(qc, backend,optimization_level=3)
            qobj = assemble(tqc,shots=shot)
            result = backend.run(qobj).result().get_counts(qc)
            for output in [bit0,bitq1,bitq2]:
                if output in result:
                    result[output]/=shot
                else:
                    result[output] = 0
            firesult=4*(result[bit0])-1-(2*(result[bit0]+result[bitq1])-1)-(2*(result[bit0]+result[bitq2])-1)# =(4P(i,j=0)-1-(2P(i=0)-1)-(2P(j=0)-1))
        elif gate == "Y":
            qc.sdg(q1) # Apply Sdg gate to transform from X basic to Y basic measurement
            qc.h(q1) # Apply H gate to apply X basic measurement
            qc.measure(q1,q1)  
            tqc = transpile(qc, backend,optimization_level=3)
            qobj = assemble(tqc,shots=shot)
            result = backend.run(qobj).result().get_counts(qc)
            if bit0 in result:
                result[bit0]/=shot
            else:
                result[bit0] = 0
            firesult=(2*(result[bit0])-1)# =(2P(i=0)-1)
        return firesult

    def Ising_Hamilton(q,h):
        Ising_Hamilton = 0
        for i in range(q):
            Zterm  = 1
            Xterm  = 1
            for j in range(q-1):
                if j == i:
                    Zterm = Zterm^Z^Z
                    Xterm  = Xterm^X^I
                elif i == (q-1) and j == (i-1):
                    Xterm = Xterm^I^X
                    Zterm = 0
                else:
                    Zterm = Zterm^I
                    Xterm = Xterm^I
            Ising_Hamilton = Ising_Hamilton + Zterm + h*Xterm
        return Ising_Hamilton

    def vqe_exact(qc,parameter,q,h,file=None,shots=20000,realbackend = Aer.get_backend('qasm_simulator')):
        qc = qc.assign_parameters(parameter)

        # backend = Aer.get_backend('statevector_simulator')
        # noise_model = NoiseModel.from_backend(realbackend)
        backend = realbackend
        # Setup cluster running
        # exc = Client(address=LocalCluster())
        # backend.set_options(executor=exc)
        # backend.set_options(max_job_size=1)

        def Res(result,p,gate):
            value = 0
            if gate == 'X':
                for index in result:
                    if index[::-1][p:(p+1)] == '0':
                        value += result[index]
            elif gate == 'ZZ':
                for index in result:
                    if index[::-1][p:(p+2)] == '00' or index[::-1][p:(p+2)] == '11':
                        value += result[index]
            return (2*value/shots -1)
            
        Z_cir =  QuantumCircuit(q)
        Z_cir.append(qc,[i for i in range(q)])
        Z_cir.measure_all()
        X_cir =  QuantumCircuit(q)
        X_cir.append(qc,[i for i in range(q)])
        X_cir.h([i for i in range(q)])
        X_cir.measure_all()
        Hamil_cir = [Z_cir,X_cir]
        
        job = execute(Hamil_cir, backend,shots=shots) 
        result = job.result().get_counts()

        Ising_Hamilton = 0
        for i in range(q):
            Xterm = Res(result[1],i,'X')
            if i < (q-1):
                Zterm = Res(result[0],i,'ZZ')
            elif i == (q-1):
                Zterm = 0
            Ising_Hamilton = Ising_Hamilton + Zterm + h*Xterm
        
        # File write
        if file != None:
            file.write(f'\n Job ID: {job.job_id()} \n ')
            file.write(f'Parameter: {parameter} \n ')
            file.write(f'Result: {result} \n ')
            file.write(f'Ising Hamilton result: {Ising_Hamilton} \n')
        return Ising_Hamilton

    # QN-SPSA metric
    def QN_SPSA(qc,parameter,gk_1,Nth_iter,beta = 0.001, eps = 0.01, realbackend = Aer.get_backend('qasm_simulator')):
        shots = 20000
        q = qc.num_qubits
        num_parameters = qc.num_parameters
        # Metric tensor
        g = np.zeros([num_parameters,num_parameters])
        # Perturbation vector
        per_vec1 = [np.random.choice([1,-1])*eps for i in range(num_parameters)]
        per_vec2 = [np.random.choice([1,-1])*eps for i in range(num_parameters)]
        # PQC
        qc1 = qc.assign_parameters(np.add(np.add(parameter,per_vec1),per_vec2))
        qc2 = qc.assign_parameters(np.add(parameter,per_vec1))
        qc3 = qc.assign_parameters(np.add(np.add(parameter,[-i for i in per_vec1]),per_vec2))
        qc4 = qc.assign_parameters(np.add(parameter,[-i for i in per_vec1]))
        qc =  qc.assign_parameters(parameter).inverse()
        # Construct expecation value
        qc1.append(qc,[_ for _ in range(q)])
        qc2.append(qc,[_ for _ in range(q)])
        qc3.append(qc,[_ for _ in range(q)])
        qc4.append(qc,[_ for _ in range(q)])
        qc1.measure_all()
        qc2.measure_all()
        qc3.measure_all()
        qc4.measure_all()
        # Execute the circuits
        backend = realbackend
        job = execute([qc1,qc2,qc3,qc4], backend,shots=shots) 
        result = job.result().get_counts()
        # Get the |000..0> state index
        index = ''
        for i in range(q):
            index += '0'
        # Get the wanted results from running backend
        fun = [0]*4
        for ind in result[0]:
            if ind == index:
                fun[0] = result[0][index]/shots
        for ind in result[1]:
            if ind == index:
                fun[1] = result[1][index]/shots
        for ind in result[2]:
            if ind == index:
                fun[2] = result[2][index]/shots
        for ind in result[3]:
            if ind == index:
                fun[3] = result[3][index]/shots
        # Calculating the shifting value
        F = [fun[0],-fun[1],-fun[2],fun[3]]
        F = sum(F)

        # QN_SPSA matrix
        for i in range(num_parameters):
            for j in range(num_parameters):
                g[i][j]=(-1/2*F)/(2*eps**4)*(per_vec1[i]*per_vec2[j]+per_vec1[j]*per_vec2[i])/2

        # The smoothing of the QN_SPSA matrix
        g = Nth_iter/(Nth_iter+1)*gk_1+1/(Nth_iter+1)*g
        # Regularization to ensure invertibility (positive semi-definite)
        g = sqrtm(g.dot(g))+beta*np.identity(num_parameters)
        g = g.real
        return g

    # FD
    def FiniteDiff_exact(q,h,qc,parameters, epsilon = 1e-3 ,l_rate=0.01, ite = 100,backend = Aer.get_backend('qasm_simulator')):
        Deriva = [[0] for i in range(qc.num_parameters)]
        Upd_para = [[0] for i in range(qc.num_parameters)]
        vqe = []
        for j in range(ite):
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]
            for i in range(len(parameters)):
                finite_diff = [0 for i in range(len(parameters))]
                finite_diff [i] = epsilon

                Dvqe[i] = (1/(2*epsilon)*(vqe_exact(qc, [fix_parameters[i] + finite_diff[i] for i in range(len(fix_parameters))],q,h,realbackend=backend)-vqe_exact(qc,[fix_parameters[i] - finite_diff[i] for i in range(len(fix_parameters))],q,h,realbackend=backend)))
            
                parameters[i]-=l_rate*Dvqe[i]
                Upd_para[i].append(parameters[i])
                Deriva[i].append(Dvqe[i])
            vqe.append(vqe_exact(qc, parameters,q,h,realbackend=backend))
        return  vqe
    
    # SPSA 
    def SPSA_exact(q,h,qc,parameters, epsilon = 0.15 ,l_rate=0.01, ite = 100, backend = Aer.get_backend('qasm_simulator')):
        Deriva = [[0] for i in range(qc.num_parameters)]
        Upd_para = [[0] for i in range(qc.num_parameters)]
        vqe = []
        for _ in range(ite):
            Dvqe = [0 for i in range(len(parameters))]
            per_vec = [np.random.choice([1,-1])*epsilon for i in range(len(parameters))]
            spsa_diff = vqe_exact(qc, np.add(parameters,per_vec),q,h,realbackend=backend)-vqe_exact(qc,np.subtract(parameters,per_vec),q,h,realbackend=backend)
            for i in range(len(parameters)):

                Dvqe[i] = 1/(2*per_vec[i])*spsa_diff
            
                parameters[i]-=l_rate*Dvqe[i]
                Upd_para[i].append(parameters[i])
                Deriva[i].append(Dvqe[i])
            vqe.append(vqe_exact(qc, parameters,q,h,realbackend=backend))
        return vqe

    # QN-SPSA + SPSA
    def SPSA_QN_SPSA(q,h,qc,parameters, epsilon = 0.01 ,l_rate=0.01, ite = 100, backend = Aer.get_backend('qasm_simulator')):
        num_parameters = qc.num_parameters
        Deriva = [[0] for i in range(num_parameters)]
        Upd_para = [[0] for i in range(num_parameters)]
        vqe = []
        # Instantiate the preceding running result
        g1 = np.zeros([num_parameters,num_parameters])
        for j in range(ite):
            Dvqe = [0 for i in range(num_parameters)]
            # Perturbation vector
            per_vec = [np.random.choice([1,-1])*epsilon for i in range(num_parameters)]
            spsa_diff = vqe_exact(qc, np.add(parameters,per_vec),q,h,realbackend=backend)-vqe_exact(qc,np.subtract(parameters,per_vec),q,h,realbackend=backend)
            for i in range(num_parameters):

                Dvqe[i] = 1/(2*per_vec[i])*spsa_diff
            
           # Get the QN_SPSA
            g = QN_SPSA(qc,parameters,g1,Nth_iter=j)
            state = True
            while state:
                try:
                    # Pseudo-inverse
                    g_1 = linalg.pinv(g)
                    state = False
                except:
                    g = QN_SPSA(qc,parameters,g1,Nth_iter=i)
            # save for the next running
            g1 = g 
            # Update papramter
            for i in range(num_parameters):
                for j in range(num_parameters):
                    parameters[i]-=l_rate*g_1[i,j]*Dvqe[j]
                Upd_para[i].append(parameters[i])
                Deriva[i].append(Dvqe[i])
            vqe.append(vqe_exact(qc, parameters,q,h,realbackend=backend))
        return vqe
    
    # Input paramters from parameter file
    with open('./Initial Parameter/Array_of_number_qubit.txt') as f:
        arraydata = f.readlines()
    with open('./Initial Parameter/Epsilon.txt') as f:
        epsilondata = f.readline()
    with open('./Initial Parameter/Iteration.txt') as f:
        iteratedata = f.readline()
    with open('./Initial Parameter/Learning_rate.txt') as f:
        lratedata = f.readline()
    with open('./Initial Parameter/Magnetic_value_h.txt') as f:
        hdata = f.readline()
    with open('./Initial Parameter/Initial_point.txt') as f:
        initpointdata = f.readline()
    arr = [] # array of qubit
    h = float(hdata) # strength magnetic field
    for i in arraydata:
        arr.append(int(i))
    # set backend
    qbackend = Aer.get_backend('qasm_simulator')
    # Setup cluster running
    exc = Client(address=LocalCluster())
    qbackend.set_options(executor=exc)
    qbackend.set_options(max_job_size=1)
    # Classical solver
    cvalue = []
    qvalue = []

    for i in arr:
        q = i
        qvalue.append(q)
        op = Ising_Hamilton(q,h) 

        # Classical solver
        w,v = np.linalg.eig(Operator(op))
        minimum=w[0]
        min_spot=0
        for i in range(1,2**q):
            if w[i]<minimum:
                min_spot=i
                minimum=w[i]                   
        groundstate = v[:,min_spot]
        cvalue.append(minimum)
    # EfficientSU2 Ansatz
    # Linear Entanglement
    # Initialization
    epsilon = float(epsilondata)
    l_rate = float(lratedata)
    numberofiteration = int(iteratedata)
    optimizer = COBYLA(maxiter=int(iteratedata))

    su2_cob = []
    su2_prs_va = []
    su2_qng_va= []
    su2_spsa_va = []
    su2_qng_spsa_va = []
    su2_fd_va =[]

    su2_cob_counts = [] 
    su2_cob_va = []
    su2_cob_pa = []
    su2_cob_devi = []
    def callback(eval_count, parameters, mean, std):  
        su2_cob_counts.append(eval_count)
        su2_cob_va.append(mean)
        su2_cob_pa.append(parameters)
        su2_cob_devi.append(std)

    for i in arr:
        q = i

        # Creat ansatz
        su2Ans = {'qubit': q , 'entanglement': 'linear','reps':1 }
        su2_ans = EfficientSU2(su2Ans['qubit'] ,entanglement=su2Ans['entanglement'], reps=su2Ans['reps'])
        su2_num_para = su2_ans.num_parameters

        # Create sub-circuit for EfficientSU2 ansatz (perculiar for each type of ansatz)
        qc = [0 for i in range(int(su2_num_para/q))]
        for k in range(int(su2_num_para/(q))):
            if k % 2 ==0:              
                qc[k] = EfficientSU2(su2Ans['qubit'], entanglement=su2Ans['entanglement'], reps=int(k/2),skip_final_rotation_layer=True)
            else:
                qc[k] = EfficientSU2(su2Ans['qubit'], entanglement=su2Ans['entanglement'], reps=int((k-1)/2),skip_final_rotation_layer=False)    
        
        # Ising Hamilton matrix operator
        op = Ising_Hamilton(q,h) 

        # VQE qiskit solver, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        vqer = VQE(su2_ans,optimizer=optimizer,quantum_instance=qbackend, initial_point=parameters,callback=callback)
        result = vqer.compute_minimum_eigenvalue(op)
        su2_cob.append(result.optimal_value)

        # SPSA, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_spsa_va.extend(SPSA_exact(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))
        
        # SPSA + QN-SPSA, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_qng_spsa_va.extend(SPSA_QN_SPSA(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))

        # FD EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_fd_va.extend(FiniteDiff_exact(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))
        
        # Parameter-shift rules, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_num_para)]
        for j in range(numberofiteration):
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]
            for i in range(len(parameters)):
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[i] = np.pi/2
                Dvqe[i] = (1/2*(vqe_exact(su2_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(su2_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
                parameters[i]-=l_rate*Dvqe[i]
            su2_prs_va.append(vqe_exact(su2_ans,parameters,q,h,realbackend=qbackend))


        # Quantum Natural Gradient Descent, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]

            # Metric tensor
            g = np.zeros([su2_num_para,su2_num_para])
            sqc = [0 for i in range(int(su2_num_para/(q)))]
            for m in range(su2_num_para):
                k = m/(q)
                if k.is_integer():
                    k = int(k)
                    # For Ry rotation gate
                    if k % 2 == 0:
                        sqc[k] =  qc[k].assign_parameters(parameters[:q*k])
                        for i in range(q):
                            for j in range(q):
                                if j == i:
                                    g[i+q*k,j+q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                                else:
                                    g[i+q*k,j+q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)
                    # For Rz rotation gate
                    else:
                        sqc[k] =  qc[k].assign_parameters(parameters[:q*(k+1)])
                        for i in range(q):
                            for j in range(q):
                                if j == i:
                                    g[i+q*k,j+q*k] = 1 - Expectation_Exact('Z',j,circ=sqc[k],realbackend=qbackend)**2
                                else:
                                    g[i+q*k,j+q*k]= Expectation_Exact('ZZ',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Z',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Z',j,circ = sqc[k],realbackend=qbackend)

            # Paramter-shift rules
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[m] = np.pi/2
                Dvqe[m] = (1/2*(vqe_exact(su2_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(su2_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
            
            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(su2_num_para):
                for j in range(su2_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            su2_qng_va.append(vqe_exact(su2_ans,parameters,q,h,realbackend=qbackend))

    # Full Entanglement
    # Initialization
    epsilon = float(epsilondata)
    l_rate = float(lratedata)
    numberofiteration = int(iteratedata)
    optimizer = COBYLA(maxiter=int(iteratedata))

    su2_cob_f = []
    su2_prs_va_f = []
    su2_qng_va_f = []
    su2_spsa_va_f = []
    su2_qng_spsa_va_f = []
    su2_fd_va_f =[]

    su2_cob_counts_f = [] 
    su2_cob_va_f = []
    su2_cob_pa_f = []
    su2_cob_devi_f = []
    def callback(eval_count, parameters, mean, std):  
        su2_cob_counts_f.append(eval_count)
        su2_cob_va_f.append(mean)
        su2_cob_pa_f.append(parameters)
        su2_cob_devi_f.append(std)

    for i in arr:
        q = i

        # Creat ansatz
        su2Ans = {'qubit': q , 'entanglement': 'full','reps':1 }
        su2_ans = EfficientSU2(su2Ans['qubit'] ,entanglement=su2Ans['entanglement'], reps=su2Ans['reps'])
        su2_num_para = su2_ans.num_parameters
        
        
        # Create sub-circuit for EfficientSU2 ansatz (perculiar for each type of ansatz)
        qc = [0 for i in range(int(su2_num_para/q))]
        for k in range(int(su2_num_para/(q))):
            if k % 2 ==0:              
                qc[k] = EfficientSU2(su2Ans['qubit'], entanglement=su2Ans['entanglement'], reps=int(k/2),skip_final_rotation_layer=True)
            else:
                qc[k] = EfficientSU2(su2Ans['qubit'], entanglement=su2Ans['entanglement'], reps=int((k-1)/2),skip_final_rotation_layer=False)

        # Ising Hamilton matrix operator
        op = Ising_Hamilton(q,h) 

        # VQE qiskit solver, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        vqer = VQE(su2_ans,optimizer=optimizer,quantum_instance=qbackend, initial_point=parameters,callback=callback)
        result = vqer.compute_minimum_eigenvalue(op)
        su2_cob_f.append(result.optimal_value)

        
        # Parameter-shift rules, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        for j in range(numberofiteration):
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]
            for i in range(len(parameters)):
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[i] = np.pi/2
                Dvqe[i] = (1/2*(vqe_exact(su2_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(su2_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
                parameters[i]-=l_rate*Dvqe[i]
            su2_prs_va_f.append(vqe_exact(su2_ans,parameters,q,h,realbackend=qbackend))

        # SPSA, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_spsa_va_f.extend(SPSA_exact(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))
        
        # SPSA + QN-SPSA, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_qng_spsa_va_f.extend(SPSA_QN_SPSA(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))

        # FD EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        su2_fd_va_f.extend(FiniteDiff_exact(q,h,su2_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))

        # Quantum Natural Gradient Descent, EfficientSU2 ansatz
        parameters = [float(initpointdata) for i in range(su2_ans.num_parameters)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]

            # Metric tensor
            g = np.zeros([su2_num_para,su2_num_para])
            sqc = [0 for i in range(int(su2_num_para/(q)))]
            for m in range(su2_num_para):
                k = m/(q)
                if k.is_integer():
                    k = int(k)
                    # For Ry rotation gate
                    if k % 2 == 0:
                        sqc[k] =  qc[k].assign_parameters(parameters[:q*k])
                        for i in range(q):
                            for j in range(q):
                                if j == i:
                                    g[i+q*k,j+q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                                else:
                                    g[i+q*k,j+q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)
                    # For Rz rotation gate
                    else:
                        sqc[k] =  qc[k].assign_parameters(parameters[:q*(k+1)])
                        for i in range(q):
                            for j in range(q):
                                if j == i:
                                    g[i+q*k,j+q*k] = 1 - Expectation_Exact('Z',j,circ=sqc[k],realbackend=qbackend)**2
                                else:
                                    g[i+q*k,j+q*k]= Expectation_Exact('ZZ',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Z',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Z',j,circ = sqc[k],realbackend=qbackend)

            # Paramter-shift rules
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[m] = np.pi/2
                Dvqe[m] = (1/2*(vqe_exact(su2_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(su2_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
            
            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(su2_num_para):
                for j in range(su2_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            su2_qng_va_f.append(vqe_exact(su2_ans,parameters,q,h,realbackend=qbackend))

    # RealAmplitude Ansatz
    # Linear Entanglement
    # Initialization
    epsilon = float(epsilondata)
    l_rate = float(lratedata)
    numberofiteration = int(iteratedata)
    optimizer = COBYLA(maxiter=int(iteratedata))

    real_cob = []
    real_prs_va = []
    real_qng_va= []
    real_spsa_va = []
    real_qng_spsa_va = []
    real_qngspsa_va = []

    real_cob_counts = [] 
    real_cob_va = []
    real_cob_pa = []
    real_cob_devi = []
    def callback(eval_count, parameters, mean, std):  
        real_cob_counts.append(eval_count)
        real_cob_va.append(mean)
        real_cob_pa.append(parameters)
        real_cob_devi.append(std)

    for i in arr:
        q = i
    
        # Create ansatz
        real_ans = RealAmplitudes(q, reps=1, entanglement='linear')
        real_num_para = real_ans.num_parameters
        real_Nrotblock_1q = int(real_num_para/((1+1)*q)) # k+1 = 1+1 with k is the reps

        # Create sub-circuit for calculating QNG
        qc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
        for k in range(int(real_num_para/(real_Nrotblock_1q*q))):
            qc[k] = RealAmplitudes(q, entanglement='linear', reps=k,skip_final_rotation_layer=True)
        
        # Ising Hamilton matrix operator
        op = Ising_Hamilton(q,h) 
        
        # VQE qiskit solver, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        vqer = VQE(real_ans,optimizer=optimizer,quantum_instance=qbackend, initial_point=parameters,callback=callback)
        result = vqer.compute_minimum_eigenvalue(op)
        real_cob.append(result.optimal_value)

        # SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        real_spsa_va.extend(SPSA_exact(q,h,real_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))
        
        # SPSA + QN-SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        real_qng_spsa_va.extend(SPSA_QN_SPSA(q,h,real_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))

        # Parameter-shift rules, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        for j in range(numberofiteration):
            Dvqe = [0 for i in range(real_num_para)]
            fix_parameters = [0 for i in range(real_num_para)]
            for i in range(real_num_para):
                fix_parameters[i] = parameters[i]
            for i in range(real_num_para):
                PSR_parameter = [0 for i in range(real_num_para)]
                PSR_parameter[i] = np.pi/2
                Dvqe[i] = (1/2*(vqe_exact(real_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(real_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
                parameters[i]-=l_rate*Dvqe[i]
            real_prs_va.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))


        # Quantum Natural Gradient Descent, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]

            # Metric tensor
            g = np.zeros([real_num_para,real_num_para])
            sqc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
            for m in range(real_num_para):
                k = m/q
                if k.is_integer():
                    k = int(k)
                    sqc[k] =  qc[k].assign_parameters(parameters[:real_Nrotblock_1q*q*k])
                    for i in range(real_Nrotblock_1q*q):
                        for j in range(real_Nrotblock_1q*q):
                            if j == i: 
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                            else:
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)

            # Paramter-shift rules
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[m] = np.pi/2
                Dvqe[m] = (1/2*(vqe_exact(real_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(real_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))

            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(real_num_para):
                for j in range(real_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            real_qng_va.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))


        # QNG+SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(real_num_para)]
            per_vec = [np.random.choice([1,-1])*epsilon for i in range(real_num_para)]
            spsa_diff = vqe_exact(real_ans, np.add(parameters,per_vec),q,h,realbackend=qbackend)-vqe_exact(real_ans,np.subtract(parameters,per_vec),q,h,realbackend=qbackend)
            
            # Metric tensor
            g = np.zeros([real_num_para,real_num_para])
            sqc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
            for m in range(real_num_para):
                k = m/q
                if k.is_integer():
                    k = int(k)
                    sqc[k] =  qc[k].assign_parameters(parameters[:real_Nrotblock_1q*q*k])
                    for i in range(real_Nrotblock_1q*q):
                        for j in range(real_Nrotblock_1q*q):
                            if j == i: 
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                            else:
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)
                
                #SPSA gradient
                Dvqe[i] = 1/(2*per_vec[i])*spsa_diff
            
            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(real_num_para):
                for j in range(real_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            real_qngspsa_va.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))

    # Full Entanglement
    # Initialization
    epsilon = float(epsilondata)
    l_rate = float(lratedata)
    numberofiteration = int(iteratedata)
    optimizer = COBYLA(maxiter=int(iteratedata))

    real_cob_f = []
    real_prs_va_f = []
    real_qng_va_f = []
    real_spsa_va_f = []
    real_qng_spsa_va_f = []
    real_qngspsa_va_f = []

    real_cob_counts_f = [] 
    real_cob_va_f = []
    real_cob_pa_f = []
    real_cob_devi_f = []
    def callback(eval_count, parameters, mean, std):  
        real_cob_counts_f.append(eval_count)
        real_cob_va_f.append(mean)
        real_cob_pa_f.append(parameters)
        real_cob_devi_f.append(std)

    for i in arr:
        q = i
        
        # Create ansatz
        real_ans = RealAmplitudes(q, reps=1, entanglement='full')
        real_num_para = real_ans.num_parameters
        real_Nrotblock_1q = int(real_num_para/((1+1)*q)) # k+1 = 1+1 with k is the reps

        # Create sub-circuit for calculating QNG
        qc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
        for k in range(int(real_num_para/(real_Nrotblock_1q*q))):
            qc[k] = RealAmplitudes(q, entanglement='linear', reps=k,skip_final_rotation_layer=True)

        # Ising Hamilton matrix operator
        op = Ising_Hamilton(q,h) 
        
        # VQE qiskit solver, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_ans.num_parameters)]
        vqer = VQE(real_ans,optimizer=optimizer,quantum_instance=qbackend, initial_point=parameters,callback=callback)
        result = vqer.compute_minimum_eigenvalue(op)
        real_cob_f.append(result.optimal_value)

        # SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_ans.num_parameters)]
        real_spsa_va_f.extend(SPSA_exact(q,h,real_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))
        
        # SPSA + QN-SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_ans.num_parameters)]
        real_qng_spsa_va_f.extend(SPSA_QN_SPSA(q,h,real_ans,parameters,ite=numberofiteration,epsilon=epsilon,l_rate=l_rate,backend=qbackend))

        # Parameter-shift rules, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_ans.num_parameters)]
        for j in range(numberofiteration):
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]
            for i in range(len(parameters)):
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[i] = np.pi/2
                Dvqe[i] = (1/2*(vqe_exact(real_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(real_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
                parameters[i]-=l_rate*Dvqe[i]
            real_prs_va_f.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))


        # Quantum Natural Gradient Descent, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_ans.num_parameters)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(len(parameters))]
            fix_parameters = [0 for i in range(len(parameters))]
            for i in range(len(parameters)):
                fix_parameters[i] = parameters[i]

            # Metric tensor
            g = np.zeros([real_num_para,real_num_para])
            sqc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
            for m in range(real_num_para):
                k = m/q
                if k.is_integer():
                    k = int(k)
                    sqc[k] =  qc[k].assign_parameters(parameters[:real_Nrotblock_1q*q*k])
                    for i in range(real_Nrotblock_1q*q):
                        for j in range(real_Nrotblock_1q*q):
                            if j == i: 
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                            else:
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)

            # Paramter-shift rules
                PSR_parameter = [0 for i in range(len(parameters))]
                PSR_parameter[m] = np.pi/2
                Dvqe[m] = (1/2*(vqe_exact(real_ans,[fix_parameters[i] + PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)-vqe_exact(real_ans,[fix_parameters[i] - PSR_parameter[i] for i in range(len(fix_parameters))],q,h,realbackend=qbackend)))
            
            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(real_num_para):
                for j in range(real_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            real_qng_va_f.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))


        # QNG+SPSA, RealAmplitude ansatz
        parameters = [float(initpointdata) for i in range(real_num_para)]
        for l in range(numberofiteration): 
            Dvqe = [0 for i in range(real_num_para)]
            per_vec = [np.random.choice([1,-1])*epsilon for i in range(real_num_para)]
            spsa_diff = vqe_exact(real_ans, np.add(parameters,per_vec),q,h,realbackend=qbackend)-vqe_exact(real_ans,np.subtract(parameters,per_vec),q,h,realbackend=qbackend)
            
            # Metric tensor
            g = np.zeros([real_num_para,real_num_para])
            sqc = [0 for i in range(int(real_num_para/(real_Nrotblock_1q*q)))]
            for m in range(real_num_para):
                k = m/q
                if k.is_integer():
                    k = int(k)
                    sqc[k] =  qc[k].assign_parameters(parameters[:real_Nrotblock_1q*q*k])
                    for i in range(real_Nrotblock_1q*q):
                        for j in range(real_Nrotblock_1q*q):
                            if j == i: 
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k] = 1 - Expectation_Exact('Y',j,circ=sqc[k],realbackend=qbackend)**2
                            else:
                                g[i+real_Nrotblock_1q*q*k,j+real_Nrotblock_1q*q*k]= Expectation_Exact('YY',i,j,circ= sqc[k],realbackend=qbackend)-Expectation_Exact('Y',i,circ= sqc[k],realbackend=qbackend)*Expectation_Exact('Y',j,circ = sqc[k],realbackend=qbackend)
                
                #SPSA gradient
                Dvqe[i] = 1/(2*per_vec[i])*spsa_diff
            
            # Pseudo-inverse
            g = linalg.pinv(g)
            # Update papramter
            for i in range(real_num_para):
                for j in range(real_num_para):
                    parameters[i]-=l_rate*g[i,j]*Dvqe[j]
            real_qngspsa_va_f.append(vqe_exact(real_ans,parameters,q,h,realbackend=qbackend))



    # Create Ising ground energy array from large energy list
    real_prs = []
    real_qng = []
    real_spsa = []
    real_qng_spsa = []
    real_qngspsa = []

    su2_prs = []
    su2_qng = []
    su2_spsa = []
    su2_qng_spsa = []
    su2_fd =[]

    su2_prs_f = []
    su2_qng_f = []
    su2_spsa_f = []
    su2_qng_spsa_f = []
    su2_fd_f = []

    real_prs_f = []
    real_qng_f = []
    real_spsa_f = []
    real_qng_spsa_f = []
    real_qngspsa_f = []
    for i in range(len(arr)):

        real_prs.append(real_prs_va[(i+1)*numberofiteration-1])
        real_qng.append(real_qng_va[(i+1)*numberofiteration-1])
        real_spsa.append(real_spsa_va[(i+1)*numberofiteration-1])
        real_qng_spsa.append(real_qng_spsa_va[(i+1)*numberofiteration-1])
        real_qngspsa.append(real_qngspsa_va[(i+1)*numberofiteration-1])

        su2_prs.append(su2_prs_va[(i+1)*numberofiteration-1])
        su2_qng.append(su2_qng_va[(i+1)*numberofiteration-1])
        su2_spsa.append(su2_spsa_va[(i+1)*numberofiteration-1])
        su2_qng_spsa.append(su2_qng_spsa_va[(i+1)*numberofiteration-1])
        su2_fd.append(su2_fd_va[(i+1)*numberofiteration-1])

        su2_prs_f.append(su2_prs_va_f[(i+1)*numberofiteration-1])
        su2_qng_f.append(su2_qng_va_f[(i+1)*numberofiteration-1])
        su2_spsa_f.append(su2_spsa_va_f[(i+1)*numberofiteration-1])
        su2_qng_spsa_f.append(su2_qng_spsa_va_f[(i+1)*numberofiteration-1])
        su2_fd_f.append(su2_fd_va_f[(i+1)*numberofiteration-1])

        real_prs_f.append(real_prs_va_f[(i+1)*numberofiteration-1])
        real_qng_f.append(real_qng_va_f[(i+1)*numberofiteration-1])
        real_spsa_f.append(real_spsa_va_f[(i+1)*numberofiteration-1])
        real_qng_spsa_f.append(real_qng_spsa_va_f[(i+1)*numberofiteration-1])
        real_qngspsa_f.append(real_qngspsa_va_f[(i+1)*numberofiteration-1])

    # Calculate the discrepancy
    for i in range(len(arr)):

        real_cob[i] -= np.real(cvalue[i])
        real_prs[i] -= np.real(cvalue[i])
        real_qng[i] -= np.real(cvalue[i])
        real_spsa[i] -= np.real(cvalue[i])
        real_qng_spsa[i] -= np.real(cvalue[i])
        real_qngspsa[i] -= np.real(cvalue[i])
        
        su2_cob[i] -= np.real(cvalue[i])
        su2_prs[i] -= np.real(cvalue[i])
        su2_qng[i] -= np.real(cvalue[i])
        su2_spsa[i] -= np.real(cvalue[i])
        su2_qng_spsa[i] -= np.real(cvalue[i])
        su2_fd[i] -= np.real(cvalue[i])

        su2_cob_f[i] -= np.real(cvalue[i])
        su2_prs_f[i] -= np.real(cvalue[i])
        su2_qng_f[i] -= np.real(cvalue[i])
        su2_spsa_f[i] -= np.real(cvalue[i])
        su2_qng_spsa_f[i] -= np.real(cvalue[i])
        su2_fd_f[i] -= np.real(cvalue[i])

        real_cob_f[i] -= np.real(cvalue[i])
        real_prs_f[i] -= np.real(cvalue[i])
        real_qng_f[i] -= np.real(cvalue[i])
        real_spsa_f[i] -= np.real(cvalue[i])
        real_qng_spsa_f[i] -= np.real(cvalue[i])
        real_qngspsa_f[i] -= np.real(cvalue[i])

    # Accuracy plot
    plt.figure()
    plt.suptitle('Deviation of Ising Ground State Energy running on Qubit')

    plt.subplot(221)
    plt.plot(qvalue, su2_cob, 'r--^',label='EffSU2_lin-Cob')
    plt.plot(qvalue, su2_prs, 'b--^',label='EffSU2_lin-PRS')
    plt.plot(qvalue, su2_fd, 'y--^',label='EffSU2_lin-FD')
    plt.plot(qvalue, su2_qng, 'c--^',label='EffSU2_lin-QNG')
    plt.plot(qvalue, su2_spsa, 'g--^',label='EffSU2_lin-SPSA')
    plt.plot(qvalue, su2_qng_spsa, 'm--^',label='EffSU2_lin-QNG_SPSA')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(loc='best',fontsize='x-small')
    plt.xticks([])
    # plt.ylim([0,0.22])
    # plt.ylabel('Deviation')

    plt.subplot(222)
    plt.plot(qvalue, real_cob, 'r-',label='RealAmp_lin-Cob')
    plt.plot(qvalue, real_prs, 'b-',label='RealAmp_lin-PRS')
    plt.plot(qvalue, real_qng, 'c-',label='RealAmp_lin-QNG')
    plt.plot(qvalue, real_spsa, 'g-',label='RealAmp_lin-SPSA')
    plt.plot(qvalue, real_qng_spsa, 'm-',label='RealAmp_lin-QNG_SPSA')
    plt.plot(qvalue, real_qngspsa, 'y-',label='RealAmp_lin-QNG+SPSA')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(loc='best',fontsize='x-small',fancybox=True, shadow=True)
    # plt.ylabel('Deviation')
    plt.xticks([])

    plt.subplot(223)
    plt.plot(qvalue, su2_cob_f, 'r--*',label='EffSU2_ful-Cob')
    plt.plot(qvalue, su2_prs_f, 'b--*',label='EffSU2_ful-PRS')
    plt.plot(qvalue, su2_fd_f, 'y--*',label='EffSU2_ful-FD')
    plt.plot(qvalue, su2_qng_f, 'c--*',label='EffSU2_ful-QNG')
    plt.plot(qvalue, su2_spsa_f, 'g--*',label='EffSU2_ful-SPSA')
    plt.plot(qvalue, su2_qng_spsa_f, 'm--*',label='EffSU2_ful-QNG_SPSA')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(loc='best',fontsize='x-small')
    plt.xticks(fontsize='xx-small')
    # plt.ylim([0,0.22])

    plt.subplot(224)
    plt.plot(qvalue, real_cob_f, 'r--',label='RealAmp_ful-Cob')
    plt.plot(qvalue, real_prs_f, 'b--',label='RealAmp_ful-PRS')
    plt.plot(qvalue, real_qng_f, 'c--',label='RealAmp_ful-QNG')
    plt.plot(qvalue, real_spsa_f, 'g--',label='RealAmp_ful-SPSA')
    plt.plot(qvalue, real_qng_spsa_f, 'm--',label='RealAmp_ful-QNG_SPSA')
    plt.plot(qvalue, real_qngspsa_f, 'y--',label='RealAmp_ful-QNG+SPSA')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(loc='best',fontsize='x-small',fancybox=True, shadow=True)
    plt.xticks(fontsize='xx-small')
    # plt.ylabel('Deviation')

    # plt.text(1.3, -0.045, 'Qubit number', ha='center', va='center', rotation='horizontal',size='large')

    plt.savefig('./Fig/NumQubit_running_accuracy.png')

    # Save data
    f = open("./Data/EffSU2_linMap_Data.txt", "w")
    f.write('Efficient SU2 Ansatz Linear Mapping \n')
    f.write('\n   1) Cobyla: \n')
    f.write(f'      su2_cob: \n')
    for i in range(len(su2_cob)):
        f.write(f'{su2_cob[i]} \n')
    f.write(f'      su2_cob_counts: \n')
    for i in range(len(su2_cob_counts)):
        f.write(f'{su2_cob_counts[i]} \n')
    f.write(f'      su2_cob_va: \n')
    for i in range(len(su2_cob_va)):
        f.write(f'{su2_cob_va[i]} \n')
    f.write('\n   2) PRS: \n')
    f.write(f'      su2_prs_va: \n')
    for i in range(len(su2_prs_va)):
        f.write(f'{su2_prs_va[i]} \n')
    f.write('\n   3) QNG: \n')
    f.write(f'      su2_qng_va: \n')
    for i in range(len(su2_qng_va)):
        f.write(f'{su2_qng_va[i]} \n')
    f.write('\n   4) SPSA: \n')
    f.write(f'      su2_spsa_va: \n')
    for i in range(len(su2_spsa_va)):
        f.write(f'{su2_spsa_va[i]} \n')
    f.write('\n   5) QNG_SPSA: \n')
    f.write(f'      su2_qng_spsa_va: \n')
    for i in range(len(su2_qng_spsa_va)):
        f.write(f'{su2_qng_spsa_va[i]} \n')
    f.write('\n   6) FD: \n')
    f.write(f'      su2_fd_va: \n')
    for i in range(len(su2_fd_va)):
        f.write(f'{su2_fd_va[i]} \n')
    f.close()

    f = open("./Data/EffSU2_fullMap_Data.txt", "w")
    f.write('Efficient SU2 Ansatz Full Mapping \n')
    f.write('\n   1) Cobyla: \n')
    f.write(f'      su2_cob_f: \n')
    for i in range(len(su2_cob_f)):
        f.write(f'{su2_cob_f[i]} \n')
    f.write(f'      su2_cob_counts_f: \n')
    for i in range(len(su2_cob_counts_f)):
        f.write(f'{su2_cob_counts_f[i]} \n')
    f.write(f'      su2_cob_va_f: \n')
    for i in range(len(su2_cob_va_f)):
        f.write(f'{su2_cob_va_f[i]} \n')
    f.write('\n   2) PRS: \n')
    f.write(f'      su2_prs_va_f: \n')
    for i in range(len(su2_prs_va_f)):
        f.write(f'{su2_prs_va_f[i]} \n')
    f.write('\n   3) QNG: \n')
    f.write(f'      su2_qng_va_f: \n')
    for i in range(len(su2_qng_va_f)):
        f.write(f'{su2_qng_va_f[i]} \n')
    f.write('\n   4) SPSA: \n')
    f.write(f'      su2_spsa_va_f: \n')
    for i in range(len(su2_spsa_va_f)):
        f.write(f'{su2_spsa_va_f[i]} \n')
    f.write('\n   5) QNG_SPSA: \n')
    f.write(f'      su2_qng_spsa_va_f: \n')
    for i in range(len(su2_qng_spsa_va_f)):
        f.write(f'{su2_qng_spsa_va_f[i]} \n')
    f.write('\n   6) FD: \n')
    f.write(f'      su2_fd_va_f: \n')
    for i in range(len(su2_fd_va_f)):
        f.write(f'{su2_fd_va_f[i]} \n')
    f.close()

    f = open("./Data/Real_linMap_Data.txt", "w")
    f.write('RealAmplitude Ansatz Linear Mapping\n')
    f.write('\n   1) Cobyla: \n')
    f.write(f'      real_cob: \n')
    for i in range(len(real_cob)):
        f.write(f'{real_cob[i]} \n')
    f.write(f'      real_cob_counts: \n')
    for i in range(len(real_cob_counts)):
        f.write(f'{real_cob_counts[i]} \n')
    f.write(f'      real_cob_va: \n')
    for i in range(len(real_cob_va)):
        f.write(f'{real_cob_va[i]} \n')
    f.write('\n   2) PRS: \n')
    f.write(f'      real_prs_va: \n')
    for i in range(len(real_prs_va)):
        f.write(f'{real_prs_va[i]} \n')
    f.write('\n   3) QNG: \n')
    f.write(f'      real_qng_va: \n')
    for i in range(len(real_qng_va)):
        f.write(f'{real_qng_va[i]} \n')
    f.write('\n   4) SPSA: \n')
    f.write(f'      real_spsa_va: \n')
    for i in range(len(real_spsa_va)):
        f.write(f'{real_spsa_va[i]} \n')
    f.write('\n   5) QNG_SPSA: \n')
    f.write(f'      real_qng_spsa_va: \n')
    for i in range(len(real_qng_spsa_va)):
        f.write(f'{real_qng_spsa_va[i]} \n')
    f.write('\n   6) QNG+SPSA: \n')
    f.write(f'      real_qngspsa_va: \n')
    for i in range(len(real_qngspsa_va)):
        f.write(f'{real_qngspsa_va[i]} \n')
    f.close()

    f = open("./Data/Real_fullMap_Data.txt", "w")
    f.write('RealAmplitude Ansatz Full Mapping \n')
    f.write('\n   1) Cobyla: \n')
    f.write(f'      real_cob_f: \n')
    for i in range(len(real_cob_f)):
        f.write(f'{real_cob_f[i]} \n')
    f.write(f'      real_cob_counts_f: \n')
    for i in range(len(real_cob_counts_f)):
        f.write(f'{real_cob_counts_f[i]} \n')
    f.write(f'      real_cob_va_f: \n')
    for i in range(len(real_cob_va_f)):
        f.write(f'{real_cob_va_f[i]} \n')
    f.write('\n   2) PRS: \n')
    f.write(f'      real_prs_va_f: \n')
    for i in range(len(real_prs_va_f)):
        f.write(f'{real_prs_va_f[i]} \n')
    f.write('\n   3) QNG: \n')
    f.write(f'      real_qng_va_f: \n')
    for i in range(len(real_qng_va_f)):
        f.write(f'{real_qng_va_f[i]} \n')
    f.write('\n   4) SPSA: \n')
    f.write(f'      real_spsa_va_f: \n')
    for i in range(len(real_spsa_va_f)):
        f.write(f'{real_spsa_va_f[i]} \n')
    f.write('\n   5) QNG_SPSA: \n')
    f.write(f'      real_qng_spsa_va_f: \n')
    for i in range(len(real_qng_spsa_va_f)):
        f.write(f'{real_qng_spsa_va_f[i]} \n')
    f.write('\n   6) QNG+SPSA: \n')
    f.write(f'      real_qngspsa_va_f: \n')
    for i in range(len(real_qngspsa_va_f)):
        f.write(f'{real_qngspsa_va_f[i]} \n')
    f.close()

    f = open("./Data/Exact_Ising_Data.txt", "w")
    f.write(f'{np.real(cvalue)} \n')
    f.close()

if __name__ == '__main__':
    main_function()