# https://github.com/gocivici/QuantumDeathMachine/blob/9d710f4f9a65552a822f301ebc292b7aa7f6769c/QuantumDeathMachine.py

#import required libraries
from Adafruit_Thermal import *
from qiskit import QuantumCircuit, execute, Aer, IBMQ
import qiskit as q
from qiskit.tools.monitor import job_monitor, backend_overview
from qiskit.providers.ibmq import least_busy
import serial
import time

#You can add remove or edit prompts from this list.
DeathList = ["Bees","Sour milk","Cuteness","Monsters from\nthe Deep","In space,\nAlone","In Sleep","Peacefully","Toilet","Furnace","Rollercoaster","Extreme cold","Bear","Heat death of\nthe universe","Goose attack","sneezing","Boat accident"]

#open serial communacition to arduino
ser = serial.Serial('/dev/ttyUSB0',9600)

#load account from disk
provider = IBMQ.load_account()

#4 qubits 4 classical bits
qc = q.QuantumCircuit(4,4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
#qc.h(4)
qc.measure([0, 1, 2, 3],[0, 1, 2, 3])

#Print all avaliable quantum computers
print("\nAll backends overview:\n")
backend_overview()

while True:
    #Read serial data coming from the arduino and print it on the console
    read_serial=ser.readline().decode('utf-8').rstrip()
    print(read_serial)
    #execute the program if finger is detected on the sensor
    if read_serial == "Image taken":
        #find the least busy quantum computer and print it.
        backend = least_busy(provider.backends(n_qubits=5, operational=True, simulator=False))
        print("\nLeast busy 5-qubit backend:", backend.name())
        #backend = provider.get_backend('ibmq_qasm_simulator')

        #number of shots for the quantum computer. We are only throwing the dice once.
        shots = 1
        #execute the circuit on the avaliable quantum computer
        job = execute(qc, backend, shots=shots, memory=True)
        #monitor the process
        job_monitor(job)
        #get the result and store it in result.
        result = job.result()
        #get the qubits from the result and store it in dice
        dice=result.get_memory(qc)
        #print the received qubits
        print("Received qubits: " + dice[0])
        #the int command here will turn the binary to a decimal
        print("Quantum dice roll result: " + str(int(dice[0],2)))
        time.sleep(4)
        #define and wake the Thermal printer
        printer = Adafruit_Thermal("/dev/serial0", 9600, timeout=5)
        printer.wake()

        #print the result
        print("Printing Result..")
        printer.justify('C')
        printer.feed(1)
        printer.setSize('L')
        printer.setSize('M')
        printer.println("This is how")
        printer.setSize('L')
        printer.println("YOU DIE")
        import gfx.arrow as arrow
        printer.printBitmap(arrow.width, arrow.height, arrow.data)
        printer.setSize("L")
        printer.println("---------------")
        printer.println(DeathList[(int(dice[0],2))])
        printer.println("---------------")
        printer.feed(1)
        printer.setSize("S")
        printer.print("Your Qubits: ")
        printer.println(dice)

        printer.feed(3)

        printer.sleep()      # Tell printer to sleep
        #printer.wake()       # Call wake() before printing again, even if reset
        printer.setDefault() # Restore printer to defaults
