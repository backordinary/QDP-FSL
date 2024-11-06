# https://github.com/Yamini2001/Quantum-Glasses/blob/fd207adfc33ed7d798cc65383de282d0f8177c1e/tutorial.py
from cgitb import text
from cmath import pi
from distutils.log import info
from email import charset
from operator import index
from sre_parse import State
from qiskit import QuantumCircuit
from qiskit.visualization import visualize_transition
#from qiskit.visualization.exceptions import VisualizationError
import qiskit
import numpy as np
import tkinter
import tkinter as Tk
from tkinter import *
from tkinter.ttk import *
from tkinter import LEFT, END, DISABLED, NORMAL
import warnings
warnings.filterwarnings('ignore')
# define window
root = tkinter.Tk()
root.title('Quantum Glasses')
# set the icon
# root.iconbitmap(default='logo.ico')
img = PhotoImage(file='./logo.ico')
root.iconphoto(True, img)

w = 400
h = 423
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
# root.tk.call('wm', 'geometry', root._w, newGeometry)
root.resizable(0, 0)  # blocking the resizing feature
# define the colors and fonts
background = '#2c94c8'
buttons = '#834558'
special_buttons = '#bc3454'
button_font = ('Arial', 18)
display_font = ('Arial', 32)

circuit = QuantumCircuit(1, 1)
circuit.x(0)
# intitalize the Quantum Circuit


def intitalize_circuit():
    """
    Intializes the Quantum Circuit
    """
    global circuit
    circuit = QuantumCircuit(1)


intitalize_circuit()
theta = 0
# define functions


def display_gate(gate_input):
    """
    Adds a corresponding gate notation in the display to track the operations.
    If the number of operation reach ten, all gate buttons are disabled.
    """
   # Insert the defined gate
    display.insert(END, gate_input)

    # check if the number of operations has reached ten,if yes,
    # disable all the gate buttons
    input_gates = display.get()
    num_gates_pressed = len(input_gates)
    list_input_gates = list(input_gates)
    search_word = ["R", "D"]
    count_double_valued_gates = [
        list_input_gates.count(i) for i in search_word]
    num_gates_pressed -= sum(count_double_valued_gates)
    gates = []
    if num_gates_pressed == 10:
        gates = [x_gate, y_gate, z_gate, Rx_gate, Ry_gate,
                 Rz_gate, s_gate, sd_gate, t_gate, td_gate, hadamard]

    for gate in gates:
        gate.config(state=DISABLED)


def clear(circuit):
    """
    Clears the display!
    Reintializes the Quantum Circuit for fresh calculation!
    Checks if the gate buttons are disabled, if so, enables the buttons
    """
    # clear the display
    display.delete(0, END)

    # reset the circuit to initial state [0>
    intitalize_circuit()

    # checks if the buttons are disabled are if so,enables them
    if x_gate['state'] == DISABLED:
        gates = [x_gate, y_gate, z_gate, Rx_gate, Ry_gate,
                 Rz_gate, s_gate, sd_gate, t_gate, td_gate, hadamard]
        for gate in gates:
            gate.config(state=NORMAL)

# define functions


def about():
    print("Checking about")
    """
    Displays the info about the project!
    """
    info = tkinter.Toplevel()
    info.title('About')
    img = PhotoImage(file='./logo.ico')
    info.tk.call('wm', 'iconphoto', root._w, img)
    # info.geometry(650*470)
    w = 650
    h = 479
    ws = info.winfo_screenwidth()
    hs = info.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    info.geometry('%dx%d+%d+%d' % (w, h, x, y))
    info.resizable(0, 0)

    text = tkinter.Text(info, height=20, width=20)
    # create label
    label = tkinter.Label(info, text="About Quantum Glasses:")
    label.config(font=("Arial", 14))

    text_to_display = """
    About: Visualization tool for Single Qubit Rotation on Bloch Sphere

    Created by: Yamini Khurana
    Created using: Python,Tkinter,Qiskit

    Info about the gate buttons and corresponding qiskit commands:
    X=flips the state of qubit -                                    circuit.x()
    Y=rotates the state vector about Y-axis -                       circuit.y()
    Z=flips the phase by PI radians -                               circuit.z()
    Rx= parameterized rotation about X axis  -                      circuit.rx()
    Ry=parameterized rotation abotu Y axis.                         circuit.ry()
    Rz=parameterized rotation about the Z axis.                     circuit.rz()
    S= rotates the state vector about Z axis by PI/2 radians-       circuit.s()
    T= rotates the state vector about Z axis by PI/4 radians -      circuit.t()
    Sd=rotates the state vector about Z axis by -PI/2 radians -     circuit.adg()
    Td=rotates the state vector about Z axis by -PI/4 radians -     circuit.tdg()
    H= creates the state of superposition -                         circuit.h()

    For Rx,Ry and Rz,
    theta(rotation_angle) allowed range in the app is [-2*PI,2*PI]

    In case of a Visualization Error, the app closes automatically.
    This indicates that visualization of your circuit is not possible.

    At a time, only ten operations can be visualized.

    """

    # Insert the text
    text.insert(END, text_to_display)
    label.pack()
    text.pack(fill='both', expand=True)
    # text.config(state='DISABLED')


# run
    info.mainloop()


def visualize_circuit(circuit, window):
    """
    visualizes the single qubit rotations corresponding to applied gates in a separate tkinter window.
    Handles any possible visualization error
    """
    try:
        visualize_transition(circuit=circuit)
    except qiskit.visualization.exceptions.VisualizationError:
        window.destroy()


def change_theta(num, window, circuit, key):
    """
    Changes the global variable theta and destroys the window 
    """
    global theta
    theta = num * np.pi
    if(key) == 'x':
        circuit.rx(theta, 0)
        theta = 0
    elif key == 'y':
        circuit.ry(theta, 0)
        theta = 0
    else:
        circuit.rz(theta, 0)
        theta = 0
    window.destroy()


def user_input(circuit, key):
    """
     Take the user input for rotation angle for parameterized
     Rotation gates Rx,Ry,Rz.
     """
    # Initalize and define the properties of window
    get_input = tkinter.Toplevel()
    get_input.title('Get Theta')
    # get_input.icombitmap(default='logo.ico')

    # get_input.geometry(360*160)
    w = 400
    h = 423
    ws = get_input.winfo_screenwidth()
    hs = get_input.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    get_input.geometry('%dx%d+%d+%d' % (w, h, x, y))
    get_input.resizable(0, 0)
    val1 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='PI/4', command=lambda: change_theta(0.25, get_input, circuit, key))
    val1.grid(row=0, column=0)

    val2 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='PI/2', command=lambda: change_theta(0.50, get_input, circuit, key))
    val2.grid(row=0, column=1)

    val3 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='PI', command=lambda: change_theta(1.0, get_input, circuit, key))
    val3.grid(row=0, column=2)

    val4 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='2*PI', command=lambda: change_theta(2.0, get_input, circuit, key))
    val4.grid(row=0, column=3, sticky='W')

    nval1 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='-PI/4', command=lambda: change_theta(-0.25, get_input, circuit, key))
    nval1.grid(row=1, column=0)

    nval2 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='-PI/2', command=lambda: change_theta(-0.50, get_input, circuit.key))
    nval2.grid(row=1, column=1)

    nval3 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='-PI', command=lambda: change_theta(-1.0, get_input, circuit, key))
    nval3.grid(row=1, column=2)

    nval4 = tkinter.Button(get_input, height=2, width=10, bg=buttons, font=(
        "Arial", 10), text='-2*PI', command=lambda: change_theta(-2.0, get_input, circuit, key))
    nval4.grid(row=1, column=3, sticky='W')

    text_object = Text(get_input, height=20, width=20, bg="light cyan")

    note = """

    GIVE THE VALUE FOR THETA
    The value has the range[-2*PI,2*PI]
    """
    text_object.grid(sticky='WE', columnspan=4)
    text_object.insert(END, note)
    get_input.mainloop()


# define layout
# define the frames
display_frame = tkinter.LabelFrame(root)
button_frame = tkinter.LabelFrame(root, bg='black')
display_frame.pack()
button_frame.pack(fill='both', expand=True)
# define the display frame layout
display = tkinter.Entry(display_frame, width=20, font=display_font,
                        bg=background, borderwidth=10, justify=LEFT)
display.pack(padx=3, pady=4)

# define the Button Frame
x_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                        text='X', command=lambda: [display_gate('x'), circuit.x(0)])
y_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                        text='Y', command=lambda: [display_gate('y'), circuit.y(0)])
z_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                        text='Z', command=lambda: [display_gate('z'), circuit.z(0)])
x_gate.grid(row=0, column=0, ipadx=45, pady=1)
y_gate.grid(row=0, column=1, ipadx=45, pady=1)
z_gate.grid(row=0, column=2, ipadx=53, pady=1)

# define the second row of buttons
Rx_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                         text='Rx', command=lambda: [display_gate('Rx'), user_input(circuit, 'x')])
Ry_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                         text='Ry', command=lambda: [display_gate('Ry'), user_input(circuit, 'y')])
Rz_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                         text='Rz', command=lambda: [display_gate('Rz'), user_input(circuit, 'z')])
Rx_gate.grid(row=1, column=0, columnspan=1, sticky='WE', pady=1)
Ry_gate.grid(row=1, column=1, columnspan=1, sticky='WE', pady=1)
Rz_gate.grid(row=1, column=2, columnspan=1, sticky='WE', pady=1)

# define the third row of buttons
s_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                        text='S', command=lambda: [display_gate('s'), circuit.s(0)])
sd_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                         text='SD', command=lambda: [display_gate('sd'), circuit.sdg(0)])
hadamard = tkinter.Button(button_frame, font=button_font, bg=buttons,
                          text='H', command=lambda: [display_gate('h'), circuit.h(0)])
s_gate.grid(row=2, column=0, columnspan=1, sticky='WE', pady=1)
sd_gate.grid(row=2, column=1, sticky='WE', pady=1)
hadamard.grid(row=2, column=2, rowspan=2, sticky='WENS', pady=1)

# define the fifth row of buttons
t_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                        text='t', command=lambda: [display_gate('T'), circuit.t(0)])
td_gate = tkinter.Button(button_frame, font=button_font, bg=buttons,
                         text='TD', command=lambda: [display_gate('TD'), circuit.tdg(0)])
t_gate.grid(row=3, column=0, sticky='WE', pady=1)
td_gate.grid(row=3, column=1, sticky='WE', pady=1)

# define the quit and visualize buttons
quit = tkinter.Button(button_frame, font=button_font,
                      bg=special_buttons, text='Quit', command=root.destroy)
visualize = tkinter.Button(button_frame, font=button_font, bg=special_buttons,
                           text='visualize', command=lambda: visualize_circuit(circuit, root))
quit.grid(row=4, column=0, columnspan=2, sticky='WE', ipadx=5, ipady=1)
visualize.grid(row=4, column=2, columnspan=1, sticky='WE', ipadx=8, pady=1)

# define the clear button
clear_button = tkinter.Button(button_frame, font=button_font,
                              bg=special_buttons, text='clear', command=lambda: clear(circuit))
clear_button.grid(row=5, column=0, columnspan=3, sticky='WE')

# define the about button
about_button = tkinter.Button(
    button_frame, font=button_font, bg=special_buttons, text='About', command=about)
about_button.grid(row=6, column=0, columnspan=3, sticky='WE')
# run the main loop
root.mainloop()
