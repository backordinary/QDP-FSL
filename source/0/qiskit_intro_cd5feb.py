# https://github.com/ascii1011/tiny_analytics_demo/blob/02047879c50db10998b1ee7a06ced0b525945f4a/builds/airflow/local/docker_compose/min_custom/airflow/assets/dags/qiskit_intro.py

"""
https://qiskit.org/documentation/intro_tutorial1.html
"""
import pendulum

from airflow import DAG
from airflow.decorators import task


with DAG(dag_id="qiskit_intro", start_date=pendulum.datetime(2022, 11, 29, tz="UTC"), schedule=None, catchup=False) as dag:

    @task()
    def test_qiskit():

        import numpy as np
        from qiskit import QuantumCircuit, transpile
        from qiskit.providers.aer import QasmSimulator
        from qiskit.visualization import plot_histogram

        # Use Aer's qasm_simulator
        simulator = QasmSimulator()

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(2, 2)

        # Add a H gate on qubit 0
        circuit.h(0)

        # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
        circuit.cx(0, 1)

        # Map the quantum measurement to the classical bits
        circuit.measure([0,1], [0,1])

        # compile the circuit down to low-level QASM instructions
        # supported by the backend (not needed for simple circuits)
        compiled_circuit = transpile(circuit, simulator)

        # Execute the circuit on the qasm simulator
        job = simulator.run(compiled_circuit, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(compiled_circuit)
        print("\nTotal count for 00 and 11 are:",counts)

    test_qiskit()