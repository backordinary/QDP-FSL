# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Gidney/Gidney_3SAT.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute  # compile  # available_backends
from qiskit.tools.visualization import plot_histogram  # plot_state,
from qiskit import Aer  # IBMQ

# from qiskit.backends.jobstatus import JOB_FINAL_STATES
# import Qconfig
# from qiskit.tools.visualization import circuit_drawer  # plot_state_qsphere


class SAT_3(object):
    def __init__(self):
        self.n = 4
        self.q = QuantumRegister(self.n, 'q')
        self.anc2 = QuantumRegister(self.n, 'anc2')
        self.anc1 = QuantumRegister(self.n, 'anc1')
        self.anc = QuantumRegister(self.n, 'anc')
        self.c = ClassicalRegister(self.n, 'c')
        self.tar = QuantumRegister(1, 'tar')
        self.qc = QuantumCircuit(self.q, self.anc1, self.anc, self.anc2,
                                 self.tar, self.c)
        # self.qc.x(self.tar[0])

    def run(self):
        for i in range(self.n):
            self.qc.x(self.q[i])
        self.qc.barrier()
        for i in range(self.n // 2):
            self.qc.h(self.q[i])
        self.qc.barrier()

        for i in range(int(self.n)):
            self.oracle()
            self.qc.barrier()
            self.diffusion_gate()
        self.qc.barrier()

        self.qc.measure(self.q, self.c)
        """run_sim"""
        # See a list of available local simulators
        # print("Local backends: ", Aer.available_backends())

        # compile and run the Quantum circuit on a simulator backend
        backend_sim = Aer.get_backend('qasm_simulator')
        job_sim = execute(self.qc, backend_sim)
        result_sim = job_sim.result()
        # Show the results
        print("simulation: ", result_sim)
        counts = result_sim.get_counts(self.qc)

        counts_l = [(i, counts[i]) for i in counts.keys()]
        print('Sim: ', sorted(counts_l, key=lambda x: x[1], reverse=True)[:5])
        plot_histogram(counts)
        # circuit_drawer(self.qc, filename="./Pictures/3SAT.png")

    def oracle(self):
        # first statment
        self.qc.ccx(self.q[0], self.q[2], self.anc1[0])
        self.qc.ccx(self.anc1[0], self.q[3], self.anc[0])
        self.qc.barrier()
        # second statement
        self.qc.x(self.q[1])
        self.qc.x(self.q[2])
        self.qc.barrier()
        self.qc.ccx(self.q[1], self.q[2], self.anc1[1])
        self.qc.ccx(self.anc1[1], self.q[3], self.anc[1])
        self.qc.barrier()
        self.qc.x(self.q[1])
        self.qc.x(self.q[2])
        self.qc.barrier()
        # third statement
        self.qc.x(self.q[0])
        self.qc.x(self.q[3])
        self.qc.barrier()
        self.qc.ccx(self.q[0], self.q[1], self.anc1[2])
        self.qc.ccx(self.anc1[2], self.q[3], self.anc[2])
        self.qc.barrier()
        self.qc.x(self.q[0])
        self.qc.x(self.q[3])
        self.qc.barrier()
        # fourth statement
        self.qc.x(self.q[1])
        self.qc.barrier()
        self.qc.ccx(self.q[0], self.q[1], self.anc1[3])
        self.qc.ccx(self.anc1[3], self.q[2], self.anc[3])
        self.qc.barrier()
        self.qc.x(self.q[1])
        self.qc.barrier()

        # hitting with z-gate
        self.qc.ccx(self.anc[0], self.anc[1], self.anc2[0])
        self.qc.ccx(self.anc2[0], self.anc[2], self.anc2[1])
        self.qc.ccx(self.anc2[1], self.anc[3], self.tar[0])
        self.qc.barrier()
        self.qc.z(self.tar[0])
        self.qc.barrier()
        self.qc.ccx(self.anc2[1], self.anc[3], self.tar[0])
        self.qc.ccx(self.anc2[0], self.anc[2], self.anc2[1])
        self.qc.ccx(self.anc[0], self.anc[1], self.anc2[0])
        self.qc.barrier()

        # uncompute
        # fourth statement
        self.qc.x(self.q[1])
        self.qc.barrier()
        self.qc.ccx(self.q[0], self.q[1], self.anc1[3])
        self.qc.ccx(self.anc1[3], self.q[2], self.anc[3])
        self.qc.barrier()
        self.qc.x(self.q[1])
        self.qc.barrier()
        # third statement
        self.qc.x(self.q[0])
        self.qc.x(self.q[3])
        self.qc.barrier()
        self.qc.ccx(self.q[0], self.q[1], self.anc1[2])
        self.qc.ccx(self.anc1[2], self.q[3], self.anc[2])
        self.qc.barrier()
        self.qc.x(self.q[0])
        self.qc.x(self.q[3])
        self.qc.barrier()
        # second statement
        self.qc.x(self.q[1])
        self.qc.x(self.q[2])
        self.qc.barrier()
        self.qc.ccx(self.q[1], self.q[2], self.anc1[1])
        self.qc.ccx(self.anc1[1], self.q[3], self.anc[1])
        self.qc.barrier()
        self.qc.x(self.q[1])
        self.qc.x(self.q[2])
        self.qc.barrier()
        # first statment
        self.qc.ccx(self.q[0], self.q[2], self.anc1[0])
        self.qc.ccx(self.anc1[0], self.q[3], self.anc[0])
        self.qc.barrier()

    def diffusion_gate(self):
        """diffusion_gate"""
        # apply hadamard gates
        for i in range(self.n):
            self.qc.h(self.q[i])

        # apply pauli-X gates
        for i in range(self.n):
            self.qc.x(self.q[i])

        self.qc.barrier()
        """Apply multi-qubit control-pauli-Z gate
        """
        if self.n > 2:
            for i in range(self.n - 1):
                if i == 0:
                    self.qc.ccx(self.q[i], self.q[i + 1], self.anc1[i])
                elif i == 1:
                    pass
                else:
                    self.qc.ccx(self.q[i], self.anc1[i - 2], self.anc1[i - 1])
            self.qc.cz(self.anc1[self.n - 3], self.q[self.n - 1])

            for i in range(self.n - 2, -1, -1):
                if i == 0:
                    self.qc.ccx(self.q[i], self.q[i + 1], self.anc1[i])
                elif i == 1:
                    pass
                else:
                    # print('i: ', i)
                    # print('q: ', len(self.q))
                    # print('anc1: ', len(self.anc1))
                    self.qc.ccx(self.q[i], self.anc1[i - 2], self.anc1[i - 1])
        else:
            # if n == 2, only need to control-z the first with the second
            self.qc.cx(self.q[0], self.anc1[0])
            self.qc.cz(self.anc1[0], self.q[1])
            self.qc.cx(self.q[0], self.anc1[0])

        self.qc.barrier()

        # apply pauli-X gates
        for i in range(self.n):
            self.qc.x(self.q[i])

        # apply hadamard gates
        for i in range(self.n):
            self.qc.h(self.q[i])

    def toffoli_n_unborrowed(self, qubits, borrowed, target):
        """toffoli_n

        :param qubits: list of indexes for global qubits
        There are n-qubits and n-2 borrowed bits
        :param borrowed: list of indexes for the borrowed bits
        :param target: index of the target qubit
        """
        if isinstance(target, int):
            target_bit = self.q[target]
        else:
            target_bit = target

        if len(qubits) <= 3:
            if len(qubits) <= 2:
                self.qc.ccx(
                    self.q[qubits[0]] if isinstance(qubits[0], int) else
                    qubits[0], self.q[qubits[1]]
                    if isinstance(qubits[1], int) else qubits[1], target_bit)
            else:
                self.qc.ccx(
                    self.q[qubits[0]] if isinstance(qubits[0], int) else
                    qubits[0], self.q[qubits[1]]
                    if isinstance(qubits[1], int) else qubits[1], target_bit)

        else:
            m = len(qubits) - 2 - 1
            print(len(qubits), m)
            self.qc.ccx(
                self.q[qubits[m + 2]] if isinstance(qubits[m + 2], int) else
                qubits[m + 2], self.q[borrowed[m]]
                if isinstance(borrowed[m], int) else borrowed[m], target_bit)
            # first forward pass
            for i in range(m - 1, -1, -1):
                if i == 0:
                    print(qubits[i])
                    self.qc.ccx(
                        self.q[qubits[i]] if isinstance(qubits[i], int) else
                        qubits[i], self.q[qubits[i + 1]] if isinstance(
                            qubits[i + 1],
                            int) else qubits[i + 1], self.q[borrowed[i]])
                else:
                    self.qc.ccx(
                        self.q[qubits[i + 2]] if isinstance(
                            qubits[i + 2],
                            int) else qubits[i + 2], self.q[borrowed[i]]
                        if isinstance(borrowed[i], int) else borrowed[i],
                        self.q[borrowed[i + 1]] if isinstance(
                            borrowed[i + 1], int) else borrowed[i + 1])
            # first backwards pass
            for i in range(1, m - 1):
                self.qc.ccx(
                    self.q[borrowed[i]]
                    if isinstance(borrowed[i],
                                  int) else borrowed[i], self.q[qubits[i + 2]]
                    if isinstance(qubits[i + 2], int) else qubits[i + 2],
                    self.q[borrowed[i + 1]]
                    if isinstance(borrowed[i + 1], int) else borrowed[i + 1])

            # second forward pass
            self.qc.ccx(
                self.q[qubits[m + 2]] if isinstance(qubits[m + 2], int) else
                qubits[m + 2], self.q[borrowed[m]]
                if isinstance(borrowed[m], int) else borrowed[m], target_bit)
            for i in range(m - 1, -1, -1):
                if i == 0:
                    self.qc.ccx(
                        self.q[qubits[i]] if isinstance(qubits[i], int) else
                        qubits[i], self.q[qubits[i + 1]] if isinstance(
                            qubits[i + 1],
                            int) else qubits[i + 1], self.q[borrowed[i]]
                        if isinstance(borrowed[i], int) else borrowed[i])
                else:
                    self.qc.ccx(
                        self.q[qubits[i + 2]] if isinstance(
                            qubits[i + 2],
                            int) else qubits[i + 2], self.q[borrowed[i]]
                        if isinstance(borrowed[i], int) else borrowed[i],
                        self.q[borrowed[i + 1]] if isinstance(
                            borrowed[i + 1], int) else borrowed[i + 1])

            # second backwards pass
            for i in range(1, m - 1):
                self.qc.ccx(
                    self.q[borrowed[i]]
                    if isinstance(borrowed[i],
                                  int) else borrowed[i], self.q[qubits[i + 2]]
                    if isinstance(qubits[i + 2], int) else qubits[i + 2],
                    self.q[borrowed[i + 1]]
                    if isinstance(borrowed[i + 1], int) else borrowed[i + 1])


if __name__ == '__main__':
    sat = SAT_3()
    sat.run()
