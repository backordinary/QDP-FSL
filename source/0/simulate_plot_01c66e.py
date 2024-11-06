# https://github.com/qBraid/presentations/blob/699c4ebf246f96d3bc50157c9ae65487a9acb966/MIT_sloan_tech_conference/python_funcs/Simulate_Plot.py
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

def SimulatePlot(circ):
    backend_sim = Aer.get_backend('qasm_simulator')
    sim = execute(circ, backend_sim, shots=1000) #how many times the quantum circuit is run.
    sim_result = sim.result() # The results from the execution are stored in 'sim_result' and can be obtained using
    counts = sim_result.get_counts(circ) #get counts
    return plot_histogram(counts)