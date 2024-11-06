# https://github.com/Thoughtscript/x_team_quantum/blob/0eebca323be8b4824a5ceb7c13090757c57d2dd5/quantum-py/pauliz.py
from qiskit import QISKitError, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

if __name__ == '__main__':

    try:

        # Prepare program
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)

        # Put qubit in superposition
        qc.z(q[0])

        # Measure quantum state
        qc.measure(q, c)

        # Execute program
        job_sim = execute(qc, "local_qasm_simulator")
        sim_result = job_sim.result()
        print("simulation: ", sim_result)
        print(sim_result.get_counts(qc))

    except QISKitError as ex:
        print('Quantum fluctuations! Eh Gad!'.format(ex))