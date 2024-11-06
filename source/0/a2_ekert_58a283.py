# https://github.com/arijitsh/quantum/blob/57812c1db9da477c08170a4aa15ddb9a13e222c5/a2-ekert.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from numpy.random import randint

verb = False


class ThirdParty:

    def __init__(self, num_bits):
        self.circuits = []
        self.num_bits = num_bits

    def create_entangled_electrons(self):
        for i in range(self.num_bits):
            qr = QuantumRegister(2, name="qr")
            cr = ClassicalRegister(4, name="cr")
            qc = QuantumCircuit(qr, cr)
            qc.x(qr[0])
            qc.x(qr[1])
            qc.h(qr[0])
            qc.cx(qr[0], qr[1])
            # qc.draw(output="mpl")
            self.circuits.append((qr, cr, qc))
        return self.circuits


class Alice:

    def __init__(self, num_bits):
        self.final_key = []
        self.num_bits = num_bits
        self.bits = []
        self.circuits = []
        self.bases = randint(3, size=num_bits)

        self.measurements = []

    def add_measure_circuit(self, circuits):
        ret_circuits = []
        for i in range(self.num_bits):
            qr = circuits[i][0]
            cr = circuits[i][1]
            a = QuantumCircuit(qr, cr, name=('a' + str(i)))

            if self.bases[i] == 0:
                # measure the spin projection of Alice's qubit onto the a_1 direction (X basis)
                a.h(qr[0])
            elif self.bases[i] == 1:
                # measure the spin projection of Alice's qubit onto the a_2 direction (W basis)
                a.s(qr[0])
                a.h(qr[0])
                a.t(qr[0])
                a.h(qr[0])
            else:
                assert (self.bases[i] == 2)
                # measure the spin projection of Alice's qubit onto the a_3 direction (standard Z basis)

            a.measure(qr[0], cr[0])
            ret_circuits.append((qr, cr, circuits[i][2] + a))

        return ret_circuits

    def declare_bases(self):
        return self.bases

    def gen_final_key(self, measures, b_bases):
        self.final_key = []
        for q in range(self.num_bits):
            if self.bases[q] == b_bases[q]:
                self.final_key.append(measures[q])
        return None

    def show_final_key(self):
        return self.final_key


class Bob:

    def __init__(self, num_bits):
        self.measurements = []
        self.final_key = []
        self.num_bits = num_bits
        self.bases = randint(3, size=num_bits)

    def add_measure_circuit(self, circuits):
        ret_circuits = []
        for i in range(self.num_bits):
            qr = circuits[i][0]
            cr = circuits[i][1]
            b = QuantumCircuit(qr, cr, name=('b' + str(i)))

            if self.bases[i] == 0:
                # measure the spin projection of Bob's qubit onto the b_1 direction (W basis)
                b.s(qr[1])
                b.h(qr[1])
                b.t(qr[1])
                b.h(qr[1])
                # measure the spin projection of Bob's qubit onto the b_2 direction (standard Z basis)
            elif self.bases[i] == 2:
                # measure the spin projection of Bob's qubit onto the b_3 direction (V basis)
                b.s(qr[1])
                b.h(qr[1])
                b.tdg(qr[1])
                b.h(qr[1])
            else:
                assert (self.bases[i] == 1)

            b.measure(qr[1], cr[1])
            ret_circuits.append(circuits[i][2] + b)

        return ret_circuits

    def declare_bases(self):
        return self.bases

    def gen_final_key(self, measures, a_bases):
        self.final_key = []
        for q in range(self.num_bits):
            if self.bases[q] == a_bases[q]:
                self.final_key.append(1 - measures[q])
        return None

    def show_final_key(self):
        return self.final_key


def generate_key():
    num_bits = 50
    third_party = ThirdParty(num_bits)

    circuits = third_party.create_entangled_electrons()
    alice = Alice(num_bits)
    bob = Bob(num_bits)

    alice_circuits = alice.add_measure_circuit(circuits)
    final_circuits = bob.add_measure_circuit(alice_circuits)

    backend = Aer.get_backend('qasm_simulator')
    result = execute(final_circuits, backend=backend, shots=1).result()

    a_measures = []
    b_measures = []
    for i in range(num_bits):
        if verb:
            print(alice.bases[i], " ", bob.bases[i])
            print(final_circuits[i].draw(output="text"))
            print(result.get_counts(final_circuits[i])
)
        d = result.get_counts(final_circuits[i])
        a_measures.append(int(list(d.keys())[0][3]))
        b_measures.append(1 - int(list(d.keys())[0][3]))

    a_bases = alice.declare_bases()
    b_bases = bob.declare_bases()

    alice.gen_final_key(a_measures, b_bases)
    bob.gen_final_key(b_measures, a_bases)

    print(alice.show_final_key(), " Key at Alice's side")
    print(bob.show_final_key(), "Key at Bob's side")


if __name__ == '__main__':
    generate_key()
