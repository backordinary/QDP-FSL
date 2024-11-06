# https://github.com/aubreycoffey/TSP-Quantum-Computing/blob/c78786a4ba3e05cd693d91c7dd78df2676f48906/quantum_algorithms/wsqaoa.py
#used - https://github.com/MarcWanner/Job-shop-Quantum-Computing/blob/main/QuantumSchedulers/QAOA/CircuitBuilders/WarmstartCircuitBuilder.py
#https://github.com/MarcWanner/Job-shop-Quantum-Computing/tree/main/QuantumSchedulers/QAOA/Preprocessors

import numpy as np
from qiskit import BasicAer
from qiskit.algorithms import QAOA
from qiskit.tools.jupyter import *
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator
from qiskit_optimization.algorithms import CplexOptimizer,MinimumEigenOptimizer
from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit.utils import QuantumInstance
from qiskit_optimization.converters import *
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import copy
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.variable import VarType




def wsqaoa(N,qubo,H,p,param_seed):
    optimizer = COBYLA(maxiter=1000, tol=0.0001)
    algorithm_globals.random_seed = 123
    quantum_instance = QuantumInstance(
        BasicAer.get_backend('qasm_simulator'),
        seed_simulator=123,
        seed_transpiler=123,
    )
    nqubits=N**2
    random.seed(param_seed)
    thept=[]
    for i in range(2*p):
        x=random.uniform(0, 1)
        if i%2==0:
            x=x*np.pi
        else:
            x=x*2*np.pi
        thept.append(x)
    #check if qubo is convex or not then adapt appropriately
    def relax_problem(problem) -> QuadraticProgram:
        """Change all variables to continuous."""
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            variable.vartype = VarType.CONTINUOUS
        return relaxed_problem
    def get_Qc(H,N):
        varlist=[]
        for i in range(0,N):
            for m in range(0,N):
                varlist.append('x_'+str(i)+'_'+str(m))
        nh=str(H)
        nh=nh.replace('-','+-')
        lish=nh.split('+')
        Q=np.zeros([len(varlist),len(varlist)])
        c=np.zeros(len(varlist))
        for i in lish:
            if '^2' in i:
                for j in range(0,len(varlist)):
                    if varlist[j] in i:
                        i=i.replace(varlist[j]+'^2','')
                        Q[j,j]=Q[j,j]+float(i)
            elif '*' in i:
                ls=[]
                for j in range(0,len(varlist)):
                    if varlist[j] in i:
                        ls.append(j)
                        i=i.replace(varlist[j],'')
                i=i.replace('*','')
                Q[ls[0],ls[1]]=Q[ls[0],ls[1]]+float(i) 
                Q[ls[1],ls[0]]=Q[ls[1],ls[0]]+float(i) 
            else:
                for j in range(0,len(varlist)):
                    if varlist[j] in i:
                        i=i.replace(varlist[j],'')
                        c[j]=c[j]+float(i)
        return Q,c,varlist
    def get_simple_convex_qp(Q, c,varlist):
        mdl = Model()
        n_qubits = Q.shape[0]
        x = [mdl.binary_var(i) for i in varlist]
        eigvals = np.linalg.eigvalsh(Q)
        u = eigvals[0]*np.ones(n_qubits)
        objective = mdl.sum([(c[i]) * x[i] for i in range(n_qubits)])
        objective += mdl.sum([(u[i]) * x[i] for i in range(n_qubits)])
        objective += mdl.sum([Q[i, j] * x[i] * x[j] for j in range(n_qubits) for i in range(n_qubits)])
        objective -= mdl.sum([u[i] * x[i] * x[i] for i in range(n_qubits)])
        mdl.minimize(objective)
        qp = from_docplex_mp(mdl)
        return qp

    def reachable_cstar(c_star: float, epsilon: float):
        return max(min(c_star, 1 - c_star), epsilon)
    
    
    optimizer = COBYLA(maxiter=1000, tol=0.0001)
    algorithm_globals.random_seed = 123
    quantum_instance = QuantumInstance(
        BasicAer.get_backend('qasm_simulator'),
        seed_simulator=123,
        seed_transpiler=123,
    )
    
    
    qp = relax_problem(QuadraticProgramToQubo().convert(qubo))
    sol = CplexOptimizer().solve(qp)
    c_stars = sol.samples[0].x
    c_test=np.array(c_stars)
    is_all_zero = np.all((c_test == 0.0))
    
    if is_all_zero==False:
        thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
        method='first'

    else:        
        method='second'
        Q,c,varlist=get_Qc(H,N)
        qp = get_simple_convex_qp(Q, c,varlist)
        for var in qp.variables:
            var.vartype = VarType.CONTINUOUS
        sol = CplexOptimizer().solve(qp)
        c_stars = sol.samples[0].x

        thetas = [2 * np.arcsin(np.sqrt(reachable_cstar(c_star,0))) for c_star in c_stars]

        
    init_qc = QuantumCircuit(N**2)
    for idx, theta in enumerate(thetas):
        init_qc.ry(theta, idx)

    
    beta = Parameter("β")
    ws_mixer = QuantumCircuit(N**2)
    for idx, theta in enumerate(thetas):
        ws_mixer.ry(-theta, idx)
        ws_mixer.rz(-2 * beta, idx)
        ws_mixer.ry(theta, idx)        
    
    
    qaoa_mes = QAOA(optimizer = optimizer, quantum_instance = quantum_instance,initial_state=init_qc,mixer=ws_mixer,reps=p, initial_point=thept)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa.solve(qubo)
    f_theta=qaoa_result.min_eigen_solver_result.optimal_point
    return qaoa_result,f_theta,method


