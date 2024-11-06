# https://github.com/crcresearch/qc/blob/2e7c824631943dac6dee0c5c8ce06da396cf44cb/ibmq/quantum_beats_ibmq_simulator.py
import math
from qiskit import execute
from ibmq.core import create_quantum_program_to_simulate_quantum_beats


if __name__ == "__main__":
    # circuit = create_quantum_program_to_simulate_quantum_beats(math.pi / 2)
    circuit = create_quantum_program_to_simulate_quantum_beats(math.pi/3)
    job_sim = execute(circuit, "local_qasm_simulator")
    sim_result = job_sim.result()
    # Execute on a quantum device
    # result_real = qp.execute(["qc"], shots=1024, max_credits=3, wait=10, timeout=240)

    # print("simulation: ", sim_result)
    # print(sim_result.get_counts(circuit)['01'])
    # print(sim_result.get_counts(circuit)['10'])

    for t in range(0, 30):
        w_larmor = 0.46  # 4.6e8 1/s as determined in the experiment

        circuit = create_quantum_program_to_simulate_quantum_beats(w_larmor * t)
        job_sim = execute(circuit, "local_qasm_simulator")
        sim_result = job_sim.result()
        singlet = sim_result.get_counts(circuit).get('01', 0)
        triplet = sim_result.get_counts(circuit).get('10', 0)
        print("%s, %s, %s" % (t, singlet, triplet))
