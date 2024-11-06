# https://github.com/reid-johnson2432/Quantum-Error-Correction/blob/0cab63850921b696d637e8d1a344b57f34c9dd54/pythonProject/main.py
from qiskit import transpile, Aer
from qiskit.tools.visualization import plot_histogram
from qrs_circuit import QRSCircuit, QuantumReedSolomon
from classical import ReedSolomon, convert_to_GF2


def main():
    rs = ReedSolomon(7, 3)
    qrs = QuantumReedSolomon(rs)
    qc = QRSCircuit(qrs, qrs.message_register, qrs.ancillia_register, qrs.classical_register)

    # Generate message
    message = rs.generate_message()
    codeword = rs.encode(message)
    quantum_codeword = qrs.convert_to_quantum(codeword)
    print('Quantum Codeword Generated: ', quantum_codeword)

    qc.initialize(quantum_codeword)
    qc.encode()
    qc.scramble_qubits(rs.ECA)

    qc.decode()
    qc.perform_measurement()
    print('Qubits measured')
    aer_sim = Aer.get_backend('aer_simulator')
    qobj = transpile(qc, aer_sim)

    print('Collecting Results')
    results = aer_sim.run(qobj).result().get_counts()

    results = {convert_to_GF2(key): value for key, value in list(results.items())}
    print(results)
    most_common_output = max(results, key=results.get)

    if most_common_output.all() == quantum_codeword.all():
        print('Corrected')

    plot_histogram(results)


main()
