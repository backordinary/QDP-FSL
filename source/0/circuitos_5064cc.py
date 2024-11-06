# https://github.com/pchagas72/PIBIC-cq/blob/311654a8c404b2766994b378a7b239c9a0bb62b2/atividade-05/variacoes/src/circuitos.py
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram


class Circuitos:
    def __init__(self):
        pass

    def simular_circuito(self, circuito: QuantumCircuit, title: str) -> None:
        simulador = Aer.get_backend('aer_simulator')
        circuito = transpile(circuito, simulador)
        resultado = simulador.run(circuito).result()
        counts = resultado.get_counts(circuito)
        display(plot_histogram(counts, title=title))

    def circuito_1(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(2, 2)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 1'))
        return qc

    def circuito_2(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(2, 2)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.cnot(1, 0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 2'))
        return qc

    def circuito_3(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(2, 2)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.cnot(1, 0, None, 0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 3'))
        return qc

    def circuito_4(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(3, 3)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.ccx(2, 1, 0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 4'))
        return qc

    def circuito_5(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(3, 3)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.cnot([2, 1], 0)
        qc.ccx(2, 1, 0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 5'))
        return qc

    def circuito_6(self, initial_state: list[int]) -> QuantumCircuit:
        qc = QuantumCircuit(3, 3)
        if len(initial_state) > 0:
            qc.x(initial_state)
        qc.x(0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        display(qc.draw(output='mpl'))
        display(self.simular_circuito(qc, 'Circuito 6'))
        return qc
