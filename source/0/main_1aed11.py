# https://github.com/algebraic-monkey/Team-19-Qmunity/blob/f6f01685855fd599032cf52ce43d8c87d1e25efb/StocQ%20Up%20UI/main.py
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.finance.applications.ising import portfolio
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.algorithms import VQE, QAOA
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.finance.data_providers import WikipediaDataProvider
from qiskit.aqua.components.optimizers import COBYLA
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
import datetime
import numpy as np
from tkinter import *
from tkinter import messagebox
import webbrowser

warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)
GREY = "#515B52"
GREEN = "#61C332"


def redirect():
    webbrowser.open("https://www.nasdaq.com/market-activity/stocks/screener")
    return


def display(output):
    messagebox.showinfo("Best Stock Option", f"The best stocks to buy are:\n{output}")


def output():
    stocks = list(symbol_entry.get().split(","))
    print(stocks)
    if len(stocks) > 0:
        is_ok = messagebox.askokcancel(stocks,
                                       f"The stocks you entered are:\n Stocks:{stocks}\n Do you want to proceed?")
        if is_ok:
            try:

                token = "************************"
                num_assets = len(stocks)
                wiki = WikipediaDataProvider(token=token,
                                             tickers=stocks,
                                             start=datetime.datetime(2017, 8, 15),
                                             end=datetime.datetime(2021, 3, 5))
                wiki.run()

                mu = wiki.get_period_return_mean_vector()
                sigma = wiki.get_period_return_covariance_matrix()
                q = 0.5  # set risk factor
                budget = 2  # set budget
                penalty = num_assets  # set parameter to scale the budget penalty term

                qubitOp, offset = portfolio.get_operator(mu, sigma, q, budget, penalty)

                # Prep for solvers
                seed = 50

                # Set up the classical optimiser
                cobyla = COBYLA()
                cobyla.set_options(maxiter=500)

                # Set up the quantum instance backend
                backend = Aer.get_backend('statevector_simulator')
                quantum_instance = QuantumInstance(backend=backend, shots=8000, seed_simulator=seed,
                                                   seed_transpiler=seed)

                qaoa_counts = []
                qaoa_values = []

                def store(counts, para, mean, std):
                    qaoa_counts.append(counts)
                    qaoa_values.append(mean)

                qaoa = QAOA(qubitOp, cobyla, 1, callback=store)
                qaoa.random_seed = seed

                qaoa_result = qaoa.run(quantum_instance)

                def index_to_selection(i, num_assets):
                    s = "{0:b}".format(i).rjust(num_assets)
                    x = np.array([1 if s[i] == '1' else 0 for i in reversed(range(num_assets))])
                    return x

                li = []

                def print_result(result):
                    selection = sample_most_likely(result.eigenstate)
                    value = portfolio.portfolio_value(selection, mu, sigma, q, budget, penalty)
                    print(list(selection))
                    answer = selection.tolist()
                    s = ""
                    for i in range(num_assets):
                        if (answer[i] == 1):
                            s = s + f"{stocks[i],}"
                    display(s)

                backend = Aer.get_backend('statevector_simulator')
                seed = 50

                cobyla = COBYLA()
                cobyla.set_options(maxiter=500)
                ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=3, entanglement='full')
                vqe = VQE(qubitOp, ry, cobyla)
                vqe.random_seed = seed

                quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

                result = vqe.run(quantum_instance)

                print_result(result)

                pass
            except FileNotFoundError:
                pass
            else:
                pass
            finally:
                symbol_entry.delete(0, END)
    else:
        messagebox.showerror("Error", "You cannot leave any of the fields empty!")

    return


# generate UI
window = Tk()
window.title("StocQ Up")
window.geometry("550x500")
window.config(padx=100, pady=50, bg="white")

bg = PhotoImage(file="bg_img.png")
canvas1 = Canvas(width=500, height=500, highlightthickness=0)
canvas1.place(x=-100, y=0)
canvas1.create_image(0, 0, image=bg, anchor="nw")

canvas = Canvas(width=170, height=157, highlightthickness=0, bg="white")
canvas.grid(row=0, column=1)
logo_img = PhotoImage(file="logo2.png")
logo = canvas.create_image(83, 77, image=logo_img)

buffer1 = Label(pady=40, text="", bg=GREY)
buffer1.grid(row=1, column=0)

symbol_label = Label(text="Stock Symbol:", bg=GREEN)
symbol_label.grid(row=2, column=0)
symbol_entry = Entry(width=33)
symbol_entry.grid(row=2, column=1)
redirect_btn = Button(text="Get Symbols", width=15, highlightthickness=0, bg=GREEN, command=redirect)
redirect_btn.grid(row=2, column=2)
out_btn = Button(text="Get Best Stock Option", width=44, highlightthickness=0, bg=GREEN, command=output)
out_btn.grid(row=3, column=1, columnspan=2)
symbol_entry.insert(0, "stock symbols separated by commas")
symbol_entry.focus()

window.mainloop()
