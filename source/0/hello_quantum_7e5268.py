# https://github.com/Thoughtscript/x_team_quantum/blob/0eebca323be8b4824a5ceb7c13090757c57d2dd5/quantum-py/hello_quantum.py
from qiskit import QISKitError, available_backends, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# https://github.com/QISKit/qiskit-sdk-py/blob/master/examples/python/hello_quantum.py

if __name__ == '__main__':

    try:
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q, c)

        print("Local backends: ", available_backends({'local': True}))

        job_sim = execute(qc, "local_qasm_simulator")
        sim_result = job_sim.result()

        print("simulation: ", sim_result)
        print(sim_result.get_counts(qc))

    except QISKitError as ex:
        print('There was an error in the circuit!. Error = {}'.format(ex))
