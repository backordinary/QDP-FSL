# https://github.com/achieveordie/Grover-AI/blob/374a5579b478002a0114aa270a7bc61c4c83fdc8/qiskit-implementation/grover.py
import math
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (NavigationToolbar2Tk, FigureCanvasTkAgg)

from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import circuit_drawer, plot_histogram

from Oracles import single_solution, nsfw_solutions


class Grover:
    """
    This is the responsible for encapsulating all elements of the circuit. It takes the following attributes:
    `n_qubits`:<int> number of qubits used to make the circuit.
    `oracle`:<qiskit.QuantumCircuit> the grover oracle that is to be used.
    `backend`: the backend to be used for measurement. Defaults to qasm_simulator.
    `shots`:<int> number of shots of the measurements that is to be performed. Defaults to 1024.
    """

    def __init__(self, nqubits, oracle, backend=Aer.get_backend('qasm_simulator'), shots=1024):
        self.n_qubits = nqubits
        self.oracle = oracle
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(self.n_qubits)

    def getOracle(self):
        """ Append the oracle provided by the caller."""
        self._circuit.append(self.oracle, [i for i in range(self.n_qubits)])

    def getAmplifier(self):
        """ Appends the amplification operator as a gate to the main circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for qubit in range(self.n_qubits):
            qc.h(qubit)
            qc.x(qubit)
        qc.h(self.n_qubits - 1)
        qc.mct(list(range(self.n_qubits - 1)), self.n_qubits - 1)
        qc.h(self.n_qubits - 1)
        for qubit in range(self.n_qubits):
            qc.x(qubit)
            qc.h(qubit)
        amp = qc.to_gate()
        amp.name = "AMP"
        self._circuit.append(amp, [i for i in range(self.n_qubits)])

    def prepareState(self):
        """ Prepare the initial states with superposition of all `n_qubits`."""
        for qubit in range(self.n_qubits):
            self._circuit.h(qubit)

    def measureResults(self):
        """ Measure the results and return the count dict for further analysis.
        (Yet to make it handle real hardware)
        """
        self._circuit.measure_all()
        return execute(self._circuit, self.backend, shots=self.shots).result().get_counts()

    def drawCircuit(self):
        """
        Using tkinter and it's respective matplotlib backend, draw how the complete circuit looks
        :return: None, displays a dialog box with the circuit, program pauses till it is open.
        """
        root = tkinter.Tk()
        root.title(f"Grover Solver with oracle:{self.oracle.name}")
        root.geometry("500x500")

        fig = Figure(figsize=(5, 5), dpi=90)
        axes = fig.add_subplot(111)
        circuit_drawer(circuit=self._circuit, output='mpl', ax=axes)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()

        root.mainloop()


def plotResults(count_dict, type_oracle):
    """
    This takes a dictionary of counts and plots a histogram to show the distribution.
    The keys received are in little-endian format and so need to convert it into integer.
    :param count_dict: <dict> with {key:value} pair as {number: number of times it showed as a measurement}
    :return: None, displays a pop up putting the program to a halt till it closes.
    """
    if type_oracle == 'single':
        count_dict = dict(zip(([sum([pow(2, i) * int(k) for i, k in enumerate(key)])
                                for key in count_dict.keys()]),
                              list(count_dict.values())))
    elif type_oracle == 'nsfw':
        count_dict = dict(
            zip(([sum([pow(2, i) * int(k) for i, k in enumerate(reversed(key))]) for key in count_dict.keys()]),
                list(count_dict.values())))

    root = tkinter.Tk()
    root.title("Plotting the results of measurement.")
    root.geometry("500x500")
    fig = Figure(figsize=(5, 5), dpi=90)
    axes = fig.add_subplot(111)

    plot_histogram(data=count_dict, bar_labels=True, ax=axes, color='#FFB000')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=1)

    frame = tkinter.Frame(root)
    frame.pack()
    close_button = tkinter.Button(frame, text="close", command=root.quit)
    close_button.pack(side=tkinter.BOTTOM)
    root.mainloop()


def solve(search_number, type_oracle):
    """
    The external method which will handle all the method calling of the `Grover` class.
    :param search_number: Number to be searched (not yet utilised completely)
    :return: None.
    """
    oracle = None
    n_qubits = 3

    # For single_solution case:
    if type_oracle == 'single':
        if search_number == 0:
            oracle = single_solution.Oracle0().getOracle()
        elif search_number == 1:
            oracle = single_solution.Oracle1().getOracle()
        elif search_number == 2:
            oracle = single_solution.Oracle2().getOracle()
        elif search_number == 3:
            oracle = single_solution.Oracle3().getOracle()
        elif search_number == 4:
            oracle = single_solution.Oracle4().getOracle()
        elif search_number == 5:
            oracle = single_solution.Oracle5().getOracle()
        elif search_number == 6:
            oracle = single_solution.Oracle6().getOracle()
        elif search_number == 7:
            oracle = single_solution.Oracle7().getOracle()

    elif type_oracle == 'nsfw':
        oracle = nsfw_solutions.OracleNSFW(search_number).getOracle()

    g_circuit = Grover(nqubits=n_qubits, oracle=oracle)
    g_circuit.prepareState()
    g_circuit.getOracle()
    g_circuit.getAmplifier()

    for i in range(int(math.sqrt(search_number))):
        g_circuit.getOracle()
        g_circuit.getAmplifier()

    # g_circuit.drawCircuit()
    plotResults(g_circuit.measureResults(), type_oracle=type_oracle)


if __name__ == '__main__':
    solve(1, type_oracle='nsfw')
