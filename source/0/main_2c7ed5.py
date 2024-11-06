# https://github.com/kindaqt/quantum-coins/blob/4f8ad4ef868e26c91380ecb0d13344447a3b9058/libs/game/main.py
from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.tools.visualization import plot_histogram, plot_state_city


class Game():
    coins = []

    def __init__(self):
        print("New game...")
        self.qc = QuantumCircuit()

    def add_coin(self, coin=None):
        self.coins.append(Coin(self.qc))
        print(self.qc.draw())

    def result(self):
        print("========================================")
        print("===              Result              ===")
        print("========================================")
        print(self.qc.draw())

        self.qc.save_statevector()

        sim = AerSimulator(method="statevector")
        simulation = sim.run(self.qc, validate=True, memory=True)
        result = simulation.result()

        counts = result.get_counts()
        print(f"counts={counts}")

        statevector = result.get_statevector()
        print(f"statevector={statevector}")

        memory = result.get_memory()
        print(f"memory={memory}")

        counts_plot_histogram = plot_histogram(
            counts, title='Bell State Counts')
        print(f"counts_plot_histogram={counts_plot_histogram}")

        statevector_plot_state_city = plot_state_city(
            statevector, title='Bell State')
        print(f"statevector_plot_state_city={statevector_plot_state_city}")

        return result


class Coin():
    def __init__(self, circuit, state_label="0", state_vector=None):
        print("Creatig new coin...")

        self.qc = circuit
        self.qubit = Qubit()
        self.clbit = Clbit()

        qr = QuantumRegister(bits=[self.qubit])
        cr = ClassicalRegister(bits=[self.clbit])
        self.qc.add_register(qr, cr)

        state_vector = state_vector or Statevector.from_label(state_label)
        self.qc.append(Initialize(state_vector), qargs=[self.qubit], cargs=[self.clbit])

        print('Created new coin:')
        print(self.qc.draw())

    def flip(self):
        print("flipping...")
        self.qc.h(self.qubit)
        # print(self.qc.draw())

    def view(self):
        print("viewing...", self.qubit, self.clbit)
        self.qc.measure(self.qubit, self.clbit)
        # print(self.qc.draw())

    def land(self):
        print("landing...")
        self.qc.h(self.qubit)
        # print(self.qc.draw())

    def touch(self, target):
        print("touching", self.qubit.name, "to", target.name)
        self.qc.cx(self.qubit, target)
        # print(self.qc.draw())


# if __name__ == '__main__':
#     game = Game()
#     game.add_coin()
#     game.add_coin()
#     # game.add_coin()
    
#     game.coins[0].flip()
#     game.coins[0].view()
#     game.result()
