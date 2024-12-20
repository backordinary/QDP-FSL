# https://github.com/Qiskit-Partners/qiskit-dell-runtime/blob/a1e5df1086fa56ac80fb7d00e7a99625728a3c05/acceptance_tests/dir_circuit_runner.py
from dell_runtime import DellRuntimeProvider
from qiskit import QuantumCircuit
import os

RUNTIME_PROGRAM_METADATA = {
    "max_execution_time": 600,
    "description": "Qiskit test program"
}

# PROGRAM_PREFIX = 'qiskit-test'

SERVER_URL = os.getenv("SERVER_URL")

def main():
    provider = DellRuntimeProvider()
    here = os.path.dirname(os.path.realpath(__file__))
    provider.remote(SERVER_URL)
    program_id = provider.runtime.upload_program(here + "/dirtest", metadata=RUNTIME_PROGRAM_METADATA)
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    program_inputs = {
        'circuits': qc,
    }

    job = provider.runtime.run(program_id, options=None, inputs=program_inputs)

    job.result(timeout=120)
    

if __name__ == "__main__":
    main()
