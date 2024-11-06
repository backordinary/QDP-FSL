# https://github.com/0xrutvij/quantum-gradient-estimation/blob/ec22b189f59bc572ae5da09e9d493d61338c9109/utils.py
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerJob, AerSimulator
from qiskit.result import Result


def run_job(
    circuit: QuantumCircuit, backend: AerSimulator, shots=1000
) -> dict[str, int]:
    transpiled_circ = transpile(circuit, backend)
    qobj = assemble(transpiled_circ, shots=shots)
    job: AerJob = backend.run(qobj)
    res: Result = job.result()
    return res.get_counts()


def float2_to_float10(number: str) -> float:
    n_sign = 1

    if number[0] == "-":
        n_sign = -1

    binnum, binfloat = number.split(".")

    decifloat = int(binnum or "0", 2)

    for i, val in enumerate(binfloat):
        decifloat += float(val) / 2 ** (i + 1)

    return decifloat * n_sign


if __name__ == "__main__":

    print(float2_to_float10("0.10011"))
