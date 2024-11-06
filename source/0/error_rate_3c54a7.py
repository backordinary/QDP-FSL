# https://github.com/vietphamngoc/QPAC/blob/e64d34d10d3e9dca993717707c5da5e8d9d70a93/error_rate.py
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import StatevectorSimulator

from oracle import Oracle
from tnn import TNN


def get_error_rate(ora: Oracle, tun_net: TNN)->float:
    """
    Function to compute the true error rate of a tunable network using the statevector simulator.

    Arguments:
        - ora: Oracle, the query oracle corresponding to the target concept
        - tun_net: TNN, the tunable network for which the error rate is to be computed

    Returns:
        - The error rate
    """
    sv_simulator = StatevectorSimulator()
    n = ora.dim
    rate = 0
    qc = QuantumCircuit(n+1)
    qc.append(ora.gate, range(n+1))
    qc.append(tun_net.network, range(n+1))

    compiled_circuit = transpile(qc, sv_simulator)
    job = sv_simulator.run(compiled_circuit)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    for s in counts:
        if s[0] == "1":
            rate += counts[s]
    return(rate)