# https://github.com/TheSleepyKing/UWCProject/blob/fdf596db4fc69b6550e1b2989da7248410d96377/main.py
#Useful packages
import PySimpleGUI as sg
from qiskit import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import randomKeys
from datetime import datetime
import tests as test
import NIST_TEST_SUITE as nist
import base64


dateTime = datetime.now()
sg.theme('Black')    
# Very basic window.
# Return values using
# automatic-numbered keys
output = sg.Text()
layout = [
    [sg.Text("Text: "),sg.Input(key='INPUT'), sg.Button('Add')],
    [sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-",file_types=(("TXT Files","*.txt"),("Text Files", "*.txt*")))],[sg.Button("Submit")],
    [sg.Button('PRNG'), sg.Button('1'), sg.Button('3'), sg.Button('5'), sg.Button('7'), sg.Button('Decrypt')],
    [sg.Checkbox('NIST test suite',default=False, key="-IN10-")],
    [sg.Text("", size=(0, 1)), sg.Input(key='OUTPUT')],
    [sg.Text("Key: "),  sg.Input(key='KEY'), sg.Button("Clear")],
    [sg.Canvas(key="-CANVAS-"),sg.Canvas(key="-CANVAS2-"),sg.Canvas(key="-CANVAS3-")],
    [sg.Exit()]
]
screenWidth, screenHeight = sg.Window.get_screen_size()
window = sg.Window("One Time Pad Application", layout)
count = 0
figg_agg1 = FigureCanvasTkAgg()
figg_agg2 = FigureCanvasTkAgg()
figg_agg3 = FigureCanvasTkAgg()

def deleteFig(figg_agg1,figg_agg2,figg_agg3):
        test.delete_figure_agg(figg_agg1)
        test.delete_figure_agg(figg_agg2)
        test.delete_figure_agg(figg_agg3)

def encrypt(key):
        shortened_key =key[:len(message)]
        print(shortened_key)
        message_encoded=''
        # chr(ord(m)+2*ord(k)%256)
        for m,k in zip(message,shortened_key):
            cipher_encode = ord(m)^ord(k)
            message_data = cipher_encode
            message_encoded += chr(message_data)
        return message_encoded, shortened_key

def decrypt(shortened_key, name):
    result =''
    for m,k in zip(shortened_key, name):
        cipher_encode = ord(m)^ord(k)
        message_data = cipher_encode
        result += chr(message_data)
    return result

def augment_message_length(message):
    """
    In order to restrict the key from generating a pattern , mainly if the key is
    shorter than the message , it would be easier to decrypt by enemies, since they
    key will use part of its own more than once generating a pattern 
    In this function , before we generate the key we intialize the key to be 
    3x greater.
    """
    #intialize size of key
    messageLength = len(message)*3
    #break up message into smaller parts if length >10
    messageLengthList= []
    for i in range(int(messageLength/10)):
        messageLengthList.append(10)
    if messageLength%10 != 0:
        messageLengthList.append(messageLength%10)

    return messageLength
while True:
# Start GUI
    """
    In this part of the application , users will be able to input text or browse for a text file containing text which make use of various number generators 
    to encrypt and decrypt.
    The inspiration for the code of the one time pad application came from a tutorial by the qiskit community
    https://github.com/qiskit-community/qiskit-community-tutorials/tree/master/awards/teach_me_qiskit_2018
    Users will have an option to apply industry standard tests namely NIST test suite which is found (https://github.com/stevenang/randomness_testsuite)
    The supporting document of this test suite is (http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf)
    """
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event=='Exit':
        break
    elif event == 'Add':
        message = values['INPUT']
        #intialize size of key
        messageLength = augment_message_length(message)
    elif event=='PRNG':
        count+ 1
        key = randomKeys.psuedo_key(messageLength)            
        name, shortened_key = encrypt(key)
        window['OUTPUT'].update(value=name)   
        window['KEY'].update(key) 
        print(name)
        randomwalk = test.randomwalktest(key)
        figg_agg1 = test.draw_figure(window['-CANVAS-'].TKCanvas, test.randwalktestVisual(randomwalk))
        digitsFrequency = test.digitsfrequencytest(key)
        figg_agg2 = test.draw_figure(window['-CANVAS2-'].TKCanvas, test.digitsfrequencytestVisual(digitsFrequency))
        figg_agg3 = test.draw_figure(window['-CANVAS3-'].TKCanvas, test.matrixVisual(key))

        if values["-IN10-"]:
            todayDate = dateTime.strftime("%d-%m-%Y_%H-%M-%S")
            g = open("PRNG_NIST_RESULTS_" + todayDate +".txt","x")
            g.close()
            f = open("PRNG_NIST_RESULTS_" + todayDate+".txt", "w")
            monobit_test = nist.monobit_test(key)
            cumulative_sums_testR = nist.cumulative_sums_test(key, 0)
            cumulative_sums_test= nist.cumulative_sums_test(key, 1)
            linear_complexity_test = nist.linear_complexity_test(key)
            random_excursions_test = nist.random_excursions_test(key)
            approximate_entropy_test =nist.approximate_entropy_test(key)
            run_test = nist.run_test(key)
            serial_test = nist.serial_test(key)
            f.write("Monobit Test Results: " + str(monobit_test))
            f.write("\n")
            f.write("Cumalative Sum Results Reverse: "+str(cumulative_sums_testR))
            f.write("\n")
            f.write("Cumalative Sum Results: "+str(cumulative_sums_test))
            f.write("\n")
            f.write("Linear Complexity Test Results: "+str(linear_complexity_test))
            f.write("\n")
            f.write("Random Execursion Test Results: "+str(random_excursions_test))
            f.write("\n")
            f.write("Approximate Entropy Test Results: "+str(approximate_entropy_test))
            f.write("\n")
            f.write("Run test Results: "+str(run_test))
            f.write("\n")
            f.write("Serial Test Result:"+str(serial_test))
            f.close()
            # print(nist.spectral_test(key))
            # print(nist.non_overlapping_test(key))
            # print(nist.statistical_test(key))



    elif event == "Submit":
        message = values["-IN-"]
        if message.endswith(".txt"):
            message = randomKeys.text_conversion(message)
        messageLength = augment_message_length(message)
        print(messageLength)   
    elif event=='1':
        key = randomKeys.Quantum_key(messageLength,1)
        name, shortened_key = encrypt(key)
        window['OUTPUT'].update(value=name)
        window['KEY'].update(key)
        if values["-IN10-"]:
            todayDate = dateTime.strftime("%d-%m-%Y_%H-%M-%S")
            g = open("QRNG_H1_NIST_RESULTS_" + todayDate+".txt","x")
            g.close()
            f = open("QRNG_H1_NIST_RESULTS_" + todayDate+".txt", "w")
            monobit_test = nist.monobit_test(key)
            cumulative_sums_testR = nist.cumulative_sums_test(key, 0)
            cumulative_sums_test= nist.cumulative_sums_test(key, 1)
            linear_complexity_test = nist.linear_complexity_test(key)
            random_excursions_test = nist.random_excursions_test(key)
            approximate_entropy_test =nist.approximate_entropy_test(key)
            run_test = nist.run_test(key)
            serial_test = nist.serial_test(key)
            f.write("Monobit Test Results: " + str(monobit_test))
            f.write("\n")
            f.write("Cumalative Sum Results Reverse: "+str(cumulative_sums_testR))
            f.write("\n")
            f.write("Cumalative Sum Results: "+str(cumulative_sums_test))
            f.write("\n")
            f.write("Linear Complexity Test Results: "+str(linear_complexity_test))
            f.write("\n")
            f.write("Random Execursion Test Results: "+str(random_excursions_test))
            f.write("\n")
            f.write("Approximate Entropy Test Results: "+str(approximate_entropy_test))
            f.write("\n")
            f.write("Run test Results: "+str(run_test))
            f.write("\n")
            f.write("Serial Test Result:"+str(serial_test))
            f.close()

        randomwalk = test.randomwalktest(key)
        figg_agg1 = test.draw_figure(window['-CANVAS-'].TKCanvas, test.randwalktestVisual(randomwalk))
        digitsFrequency = test.digitsfrequencytest(key)
        figg_agg2= test.draw_figure(window['-CANVAS2-'].TKCanvas, test.digitsfrequencytestVisual(digitsFrequency))
        figg_agg3 = test.draw_figure(window['-CANVAS3-'].TKCanvas, test.matrixVisual(key))

    elif event=='3':
        count+ 1
        key = randomKeys.Quantum_key(messageLength,3)
        name, shortened_key = encrypt(key)
        window['OUTPUT'].update(value=name)
        window['KEY'].update(key)
        randomwalk = test.randomwalktest(key)
        figg_agg1 = test.draw_figure(window['-CANVAS-'].TKCanvas, test.randwalktestVisual(randomwalk))
        digitsFrequency = test.digitsfrequencytest(key)
        figg_agg2 = test.draw_figure(window['-CANVAS2-'].TKCanvas, test.digitsfrequencytestVisual(digitsFrequency))
        figg_agg3 = test.draw_figure(window['-CANVAS3-'].TKCanvas, test.matrixVisual(key))
        if values["-IN10-"]:
            todayDate = dateTime.strftime("%d-%m-%Y_%H-%M-%S")
            g = open("QRNG_H3_NIST_RESULTS_" + todayDate+".txt","x")
            g.close()
            f = open("QRNG_H3_NIST_RESULTS_" + todayDate+".txt","w")
            monobit_test = nist.monobit_test(key)
            cumulative_sums_testR = nist.cumulative_sums_test(key, 0)
            cumulative_sums_test= nist.cumulative_sums_test(key, 1)
            linear_complexity_test = nist.linear_complexity_test(key)
            random_excursions_test = nist.random_excursions_test(key)
            approximate_entropy_test =nist.approximate_entropy_test(key)
            run_test = nist.run_test(key)
            serial_test = nist.serial_test(key)
            f.write("Monobit Test Results: " + str(monobit_test))
            f.write("\n")
            f.write("Cumalative Sum Results Reverse: "+str(cumulative_sums_testR))
            f.write("\n")
            f.write("Cumalative Sum Results: "+str(cumulative_sums_test))
            f.write("\n")
            f.write("Linear Complexity Test Results: "+str(linear_complexity_test))
            f.write("\n")
            f.write("Random Execursion Test Results: "+str(random_excursions_test))
            f.write("\n")
            f.write("Approximate Entropy Test Results: "+str(approximate_entropy_test))
            f.write("\n")
            f.write("Run test Results: "+str(run_test))
            f.write("\n")
            f.write("Serial Test Result:"+str(serial_test))
            f.close()


    elif event=='5':
        count+ 1
        key = randomKeys.Quantum_key(messageLength,5)
        name, shortened_key = encrypt(key)
        window['OUTPUT'].update(value=name)
        window['KEY'].update(key)
        randomwalk = test.randomwalktest(key)
        figg_agg1 = test.draw_figure(window['-CANVAS-'].TKCanvas, test.randwalktestVisual(randomwalk))
        digitsFrequency = test.digitsfrequencytest(key)
        figg_agg2 = test.draw_figure(window['-CANVAS2-'].TKCanvas, test.digitsfrequencytestVisual(digitsFrequency))
        figg_agg3 = test.draw_figure(window['-CANVAS3-'].TKCanvas, test.matrixVisual(key))
        if values["-IN10-"]:
            todayDate = dateTime.strftime("%d-%m-%Y_%H-%M-%S")
            g = open("QRNG_H5_NIST_RESULTS_" + todayDate+".txt","x")
            g.close()
            f = open("QRNG_H5_NIST_RESULTS_" + todayDate+".txt","w")
            monobit_test = nist.monobit_test(key)
            cumulative_sums_testR = nist.cumulative_sums_test(key, 0)
            cumulative_sums_test= nist.cumulative_sums_test(key, 1)
            linear_complexity_test = nist.linear_complexity_test(key)
            random_excursions_test = nist.random_excursions_test(key)
            approximate_entropy_test =nist.approximate_entropy_test(key)
            run_test = nist.run_test(key)
            serial_test = nist.serial_test(key)
            f.write("Monobit Test Results: " + str(monobit_test))
            f.write("\n")
            f.write("Cumalative Sum Results Reverse: "+str(cumulative_sums_testR))
            f.write("\n")
            f.write("Cumalative Sum Results: "+str(cumulative_sums_test))
            f.write("\n")
            f.write("Linear Complexity Test Results: "+str(linear_complexity_test))
            f.write("\n")
            f.write("Random Execursion Test Results: "+str(random_excursions_test))
            f.write("\n")
            f.write("Approximate Entropy Test Results: "+str(approximate_entropy_test))
            f.write("\n")
            f.write("Run test Results: "+str(run_test))
            f.write("\n")
            f.write("Serial Test Result:"+str(serial_test))
            f.close()

    elif event=='7':
        count+ 1
        key = randomKeys.Quantum_key(messageLength,7)
        name, shortened_key = encrypt(key)
        window['OUTPUT'].update(value=name)
        window['KEY'].update(key)
        randomwalk = test.randomwalktest(key)
        figg_agg1 = test.draw_figure(window['-CANVAS-'].TKCanvas, test.randwalktestVisual(randomwalk))
        digitsFrequency = test.digitsfrequencytest(key)
        figg_agg2 = test.draw_figure(window['-CANVAS2-'].TKCanvas, test.digitsfrequencytestVisual(digitsFrequency))
        figg_agg3 = test.draw_figure(window['-CANVAS3-'].TKCanvas, test.matrixVisual(key))
        if values["-IN10-"]:
            todayDate = dateTime.strftime("%d-%m-%Y_%H-%M-%S")
            g = open("QRNG_H7_NIST_RESULTS_" + todayDate+".txt","x")
            g.close()
            f = open("QRNG_H7_NIST_RESULTS_" + todayDate+".txt","w")
            monobit_test = nist.monobit_test(key)
            cumulative_sums_testR = nist.cumulative_sums_test(key, 0)
            cumulative_sums_test= nist.cumulative_sums_test(key, 1)
            linear_complexity_test = nist.linear_complexity_test(key)
            random_excursions_test = nist.random_excursions_test(key)
            approximate_entropy_test =nist.approximate_entropy_test(key)
            run_test = nist.run_test(key)
            serial_test = nist.serial_test(key)
            f.write("Monobit Test Results: " + str(monobit_test))
            f.write("\n")
            f.write("Cumalative Sum Results Reverse: "+str(cumulative_sums_testR))
            f.write("\n")
            f.write("Cumalative Sum Results: "+str(cumulative_sums_test))
            f.write("\n")
            f.write("Linear Complexity Test Results: "+str(linear_complexity_test))
            f.write("\n")
            f.write("Random Execursion Test Results: "+str(random_excursions_test))
            f.write("\n")
            f.write("Approximate Entropy Test Results: "+str(approximate_entropy_test))
            f.write("\n")
            f.write("Run test Results: "+str(run_test))
            f.write("\n")
            f.write("Serial Test Result:"+str(serial_test))
            f.close()

    elif event == "Decrypt":
        name = decrypt(shortened_key, name)
        print(name)
        window['OUTPUT'].update(value=name)

    elif event =="Clear":
        deleteFig(figg_agg1,figg_agg2,figg_agg3)

window.close()


