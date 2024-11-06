# https://github.com/crcresearch/qc/blob/2e7c824631943dac6dee0c5c8ce06da396cf44cb/ibmq/quantum_beats_simulator_qiskit0_8_1.py
import math
from qiskit import Aer, execute
from ibmq.core_qiskit0_8_1 import create_quantum_circuit_to_simulate_quantum_beats


if __name__ == "__main__":
    # circuit = create_quantum_program_to_simulate_quantum_beats(math.pi / 2)
    #circuit = create_quantum_program_to_simulate_quantum_beats(math.pi/3)
    #job_sim = execute(circuit, "local_qasm_simulator")
    #sim_result = job_sim.result()
    # Execute on a quantum device
    # result_real = qp.execute(["qc"], shots=1024, max_credits=3, wait=10, timeout=240)

    # print("simulation: ", sim_result)
    # print(sim_result.get_counts(circuit)['01'])
    # print(sim_result.get_counts(circuit)['10'])

    for t in range(0, 30):
        w_larmor = 0.46  # 4.6e8 1/s as determined in the experiment

        circuit = create_quantum_circuit_to_simulate_quantum_beats(w_larmor * t)
        #job_sim = execute(circuit, "local_qasm_simulator")
        simulator = Aer.get_backend('qasm_simulator')
        #sim_result = job_sim.result()
        sim_result = execute(circuit, simulator).result()
        singlet = sim_result.get_counts(circuit).get('01', 0)
        triplet = sim_result.get_counts(circuit).get('10', 0)
        print("%s, %s, %s" % (t, singlet, triplet))
