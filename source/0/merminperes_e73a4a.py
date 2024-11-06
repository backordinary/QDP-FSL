# https://github.com/seanmcilvane/BenchmarkQuantumHardware/blob/5badb1389084e081671380fba596c4e25bc3d4de/MerminPeres.py
from qiskit import *
from qiskit import QuantumCircuit
from collections import Counter

def MerminPeresCircuit(X, Y):
    
    qc = QuantumCircuit(4)

    qc.h(0)
    qc.h(1)
    qc.cx(0, 2)
    qc.cx(1, 3)

    # Select measurement operator based on and 
    if X == 0:
        qc.z(1)
        
        qc.z(0)
        
        qc.z(0)
        qc.z(1)
      
            
    if X == 1:
        qc.x(0)
        
        qc.x(1)
            
        qc.x(0)
        qc.x(1)
        
    if X == 2:
        qc.z(0)
        qc.x(0)
        qc.z(0)
        qc.z(1)
        
        qc.x(0)
        qc.z(0)
        qc.x(0)
        qc.x(1)
        
        qc.y(0)
        qc.y(1)
    
    if Y == 0:
        qc.z(3)
        
        qc.x(2)
        
        qc.z(2)
        qc.x(2)
        qc.z(2)
        qc.z(3)
        
    if Y == 1:
        qc.z(2)
        
        qc.x(3)
        
        qc.x(2)
        qc.z(2)
        qc.x(2)
        qc.x(3)
        
    if Y == 2:
        qc.z(2)
        qc.z(3)
        
        qc.x(2)
        qc.x(3)
        
        qc.y(2)
        qc.y(3)
        
    # Measure in bell basis
    qc.cx(0, 2)
    qc.cx(1, 3)
        
    qc.h(0)
    qc.h(1)
        
    qc.measure_all()
    
    return qc



def classical_simulation_mermin_peres(backend = Aer.get_backend('qasm_simulator'), shots = 1024):
    
    total_counts = {}                                  
    
    qc = MerminPeresCircuit(0, 0)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(0, 1)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(0, 2)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(1, 0)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(1, 1)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(1, 2)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(2, 0)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(2, 1)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    
    qc = MerminPeresCircuit(2, 2)
        
    t_qc = transpile(qc, backend)

    # Run and get counts
    
    job_sim = backend.run(t_qc, shots = shots)
    
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    total_counts = Counter(total_counts) + Counter(counts)
    return total_counts