# https://github.com/lionelchg/LesAtomesDeSavoie/blob/9e0c37ce2ea634446b0406ceea90d9432461ae86/qiskit/grover/run.py
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

def run_ibmq(circuit:QuantumCircuit, figname:str) -> None:
    """ Run circuit on IBMQ

    :param circuit: Circuit object
    :type circuit: QuantumCircuit
    :param figname: name of the figure
    :type figname: str
    """
    # Load IBM Q account and get the least busy backend device
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider("ibm-q")
    
    device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and 
                                    not x.configuration().simulator and x.status().operational==True))
    print("Running on current least busy device: ", device)

    # Run our circuit on the least busy backend. Monitor the execution of the job in the queue
    transpiled_circuit = transpile(circuit, device, optimization_level=3)
    qobj = assemble(transpiled_circuit)
    job = device.run(qobj)
    job_monitor(job, interval=2)

    # Get the results from the computation
    results = job.result()
    answer = results.get_counts(grover_circuit)
    fig = plot_histogram(answer)
    fig.savefig(figname)

def run_sim(circuit:QuantumCircuit, figname:str) -> None:
    """ Run circuit on qasm simulator

    :param circuit: Circuit object
    :type circuit: QuantumCircuit
    :param figname: name of the figure
    :type figname: str
    """
    qasm_sim = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, qasm_sim)
    qobj = assemble(transpiled_circuit)
    results = qasm_sim.run(qobj).result()
    counts = results.get_counts()
    fig = plot_histogram(counts)
    fig.savefig(figname)