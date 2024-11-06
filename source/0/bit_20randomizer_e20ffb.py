# https://github.com/2000maor/Quantum-Crypto/blob/eb82d67f05181f7071018b20aae74438a612da7c/bit%20randomizer.py
# authors: Maor Marcus & Roy Mazuz
import xlsxwriter
import string
from qiskit import (execute, Aer)
from qiskit import QuantumCircuit


FILENAME = 'random1.xlsx'  # the file we write our data to
LENGTH = 100  # length of the message
MESSAGE = 'QM'  # the message that Alice want to send
ALPHABET = list(string.ascii_lowercase)
PEOPLE = ["Alice", "Eve", "Bob", "Bits"]


def true_random():
    """ creating a real randomizer in order to choose the basis """
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    return max(counts, key=counts.get)
counter = 0
for 0 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 1 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 2 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 3 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 4 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 5 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 6 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 7 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 8 in range(1000):
    counter += int(true_random())
print(counter)
counter = 0
for 9 in range(1000):
    counter += int(true_random())
print(counter)


def base26_to_binary(word):
    """ gets a word in base 26, translates to binary """
    message = ''
    letters = [char for char in word]
    for x in range(len(letters)):
        dec_code = ALPHABET.index(letters[x].lower())
        bin_code = format(dec_code, 'b')
        message += bin_code.zfill(5)
    return message


message = base26_to_binary(MESSAGE)
print("How to right QM in binary?", message)


def base_randomizer():
    """ randomize (in quantum terms) 52 bases that will be used in the BB84 key exchange protocol """
    aeb_arr = ["", "", "", ""]
    for x in range(LENGTH):
        for y in range(len(aeb_arr)):
            bit = true_random()
            if y < 3:
                if bit == '0':
                    aeb_arr[y] += '+'
                else:
                    aeb_arr[y] += 'x'
            else:
                if bit == '0':
                    aeb_arr[y] += '0'
                else:
                    aeb_arr[y] += '1'
    return aeb_arr


all_bases = base_randomizer()
print(all_bases)


def extract_data_to_excel(bases_arr):
    """ gets the bases and extracts them to an excel sheet as a chart """
    workbook = xlsxwriter.Workbook(FILENAME)
    worksheet = workbook.add_worksheet()
    for x in range(len(bases_arr)):
        worksheet.write(0, x, PEOPLE[x])  # the names of the people that chose the bases
        bases = [char for char in bases_arr[x]]
        for y in range(len(bases)):
            worksheet.write(y + 1, x, bases[y])  # Writes the base to the excel sheet
    workbook.close()


extract_data_to_excel(all_bases)
