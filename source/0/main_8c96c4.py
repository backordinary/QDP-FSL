# https://github.com/myounatan/tt-lite/blob/9ef64f26f166a49b582f2d7974710fcf4f8c51fc/src/main.py
'''
Matthew Younatan
copyright MIT License

main.py

https://arxiv.org/abs/1912.12599
'''

#------------------------#
# Imports
#------------------------#
from math import log, ceil
from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister, register

from ternarytree import Minimize as MinimizeESOP
import utils
import Qconfig

#------------------------#
# Test Constants
#------------------------#
IMAGE_FILENAME  = '../test_images/neqr/c.png' # image to load and test
IGNORE_ALPHA    = False # ignores alpha color channel (transparency)

WRITE_CSV = False # whether to create a csv file of the test results or not
CSV_FILE_NAME = 'neqr_testcases_csv' # file name for the csv file saved in ./test_results

RUN_QISKIT = False # toggle if you want QISkit to run the resulting circuit (should always be False unless you know what you're doing!)



#--------------------------------------------------------------------------------#
# DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING!
#--------------------------------------------------------------------------------#



#------------------------#
# Fixed Constants
#------------------------#
BACKEND = 'local_qasm_simulator'
BLACK = 0


#------------------------#
# Image Variables
#------------------------#
pixel_array, image_properties = GetPNGData(IMAGE_FILENAME)

image_bitdepth = image_properties['bitdepth']
isgreyscale = image_properties['greyscale']
isalpha = image_properties['alpha']
isrgb = False
image_size = image_properties['size']
image_width = image_size[0]
image_height = image_size[1]
total_pixels = image_width * image_height

color_channel_size = 1  # grey scale is 1, RGB is 3, RGBA is 4
if not isgreyscale:
    if isalpha:  # is RGBA
        color_channel_size = 4
    else:  # is RGB
        color_channel_size = 3
    isrgb = True
else:
    if isalpha:
        color_channel_size = 2  # is greyscale and alpha

print('color channel size %d\n' % color_channel_size)

COLOR_FORMAT = '0' + str(image_bitdepth) + 'b'
print('color format code: %s\n' % COLOR_FORMAT)

print("pixel_array:\n{}".format(pixel_array))


#------------------------#
# Helpers
#------------------------#
def GetColorChannelSize():
    ''' GetColorChannelSize

    returns the correct color channel size, checks IGNORE_ALPHA
    '''
    return (IGNORE_ALPHA and isalpha) and color_channel_size - 1 or color_channel_size

def GetLocationColor(pixel_num, col=None, row=None):
    ''' GetLocationColor

    from a pixel, grab its color tuple
    '''
    if not row:
        row = pixel_num // image_width
    if not col:
        col = pixel_num % image_width
        col = col * color_channel_size
    
    try:
        color = ()
        for pixel in range(col, col + (GetColorChannelSize())):
            color += (pixel_array[row][pixel],)

        #print(color)
        return color
    except IndexError:
        return None

def ColorToBinary(color):
    ''' ColorToBinary

    formats a color tuple to binary string tuple
    '''
    binary_tup = ()
    for channel in color:
        binary_tup += (format(channel, COLOR_FORMAT),)
    return binary_tup


#------------------------#
# Quantum Variables
#------------------------#
q = GetColorChannelSize() * image_bitdepth # number of qubits needed to represent pixel color data
h = ceil(log(image_size[1], 2))  # log (2) H, qubits for Y
w = ceil(log(image_size[0], 2))  # log (2) W, qubits for X
print('q = %d, h = %d, w = %d\n' % (q, h, w))

LOC_H_FORMAT = '0' + str(h) + 'b'
LOC_W_FORMAT = '0' + str(w) + 'b'
print('location format code: H: %s W: %s\n' % (LOC_H_FORMAT, LOC_W_FORMAT))

def PixelToLocationBinary(pixel_num):
    ''' PixelToQLOCBinary

    formats a pixel num integer to binary string
    '''
    row = pixel_num // image_width
    col = pixel_num % image_width

    return format(row, LOC_H_FORMAT) + format(col, LOC_W_FORMAT)

# super position of the location qubits introduces some redundancies in the location register that need to be
# taken care of when processing image location codes
QMAX_H = (2**h)
QMAX_W = (2**w)

# classic vs quantum dimension difference
CQ_HEIGHT_DIF = QMAX_H - image_height
CQ_WIDTH_DIF = QMAX_W - image_width


#------------------------#
# Toffoli Compression
#------------------------#
old_num_toffolis = 0 # before compression
num_toffolis = 0 # after compression

COLOR_QREG = QuantumRegister(q, 'color')
LOCATION_QREG = QuantumRegister(h + w, 'location')
MEASURE_CREG = ClassicalRegister(h + w + q, 'measurement')
REGS = [COLOR_QREG, LOCATION_QREG, MEASURE_CREG]

if (h + w) > 2:  # not the simple case (requires an ancilla register)
    ANCILLA_QREG = QuantumRegister(h + w - 1, 'ancilla')
    REGS.append(ANCILLA_QREG)
    #dag_initial.add_qreg('ancilla', h + w - 1)

qcircuit = QuantumCircuit(*REGS, name='image')

qcircuit.h(LOCATION_QREG)

# this variable is used to track the X gates, so we don't have two X gates beside each other
# 0 means there is no x gate to run off of, 1 means there is
xgates = [0 for i in range(h+w)]

# iterate by color channel
total_time = 0
for channel_i in range(0,GetColorChannelSize()):
    # iterate by colorbit
    for Ci in range(0, image_bitdepth):
        controls = []#'.i %d\n.o %d\n' % (h+w, 1) # PLA format (https://ddd.fit.cvut.cz/prj/TT-Min/pla.html)
        target = 2**(Ci)
        print(bin(target))
        pixel_num = 0
        
        # iterate through pixels
        while pixel_num < total_pixels:
        
            # grab colors
            color = GetLocationColor(pixel_num)
            color_binary = ColorToBinary(color)
            
            colorbit = color_binary[channel_i][Ci]
            
            if colorbit == '1':
                old_num_toffolis += 1
                
                target_binary = format(target, COLOR_FORMAT)
                control_binary = PixelToLocationBinary(pixel_num)
                
                print(control_binary, color_binary[channel_i], 'looking at', target_binary)
                
                controls.append(control_binary) #+= '%s %s\n' % (control_binary, '1')
                
            pixel_num += 1
        
        print(controls)

        if len(controls) > 1:
            lines, time_taken = MinimizeESOP(a=controls)
            total_time += time_taken
            print(lines)
            for i, line in enumerate(lines):
                if line != '' and line.count('-') != len(line):
                    line=line[::-1]
                    num_toffolis += 1
                    
                    # create a toffoli
                    control_lines = []
                    for i, c in enumerate(line):
                        if c != '-':
                            control_lines.insert(0, i)
                            
                            if c == '0':
                                if xgates[i] == 0: # need to add an xgate, and mark it
                                    xgates[i] = 1
                                    qcircuit.x(LOCATION_QREG[i])
                                # elif xgates[i] == 1: # don't need to do anything
                            elif c == '1':
                                if xgates[i] == 1: # reset marker, add x gate
                                    xgates[i] = 0
                                    qcircuit.x(LOCATION_QREG[i]) # "2nd" x gate
                            
                    control_size = len(control_lines)
                
                    if control_size == 1:
                        qcircuit.cx(LOCATION_QREG[control_lines[0]], COLOR_QREG[Ci])
                    elif control_size == 2:
                        qcircuit.ccx(LOCATION_QREG[control_lines[0]], LOCATION_QREG[control_lines[1]], COLOR_QREG[Ci])
                    else: # control_size > 2
                        # compute
                        qcircuit.ccx(LOCATION_QREG[control_lines[0]], LOCATION_QREG[control_lines[1]], ANCILLA_QREG[0])
                        for i in range(2, control_size):
                            qcircuit.ccx(LOCATION_QREG[control_lines[i]], ANCILLA_QREG[i - 2], ANCILLA_QREG[i - 1])
                        
                        # copy
                        qcircuit.cx(ANCILLA_QREG[control_size - 2], COLOR_QREG[Ci])
            
                        # uncompute
                        for i in range(control_size - 1, 1, -1):
                            qcircuit.ccx(LOCATION_QREG[control_lines[i]], ANCILLA_QREG[i - 2], ANCILLA_QREG[i - 1])
                        qcircuit.ccx(LOCATION_QREG[control_lines[0]], LOCATION_QREG[control_lines[1]], ANCILLA_QREG[0])

# add X gates to any remaining flags
for i, x in enumerate(xgates):
    if x == 1:
        xgates[i] = 0
        qcircuit.x(LOCATION_QREG[i]) # "2nd" x gate

qcircuit.barrier()
for i in range(q):
    qcircuit.measure(COLOR_QREG[i], MEASURE_CREG[i])
for i in range(h + w):
    qcircuit.measure(LOCATION_QREG[i], MEASURE_CREG[q+i])

print(qcircuit.qasm())
print('qcircuit length:', len(qcircuit))

print('old toffoli count: %d' % old_num_toffolis)
print('new toffoli count: %d' % num_toffolis)

print('compression ratio:', 1 - (num_toffolis/old_num_toffolis))

print('time taken to compress:', total_time)


# write to csv
if WRITE_CSV:
    import csv
    with open('{}/{}.csv'.format('./test_results', CSV_FILE_NAME), 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'initial terms', 'minimized terms', 'ratio', 'time taken', 'qcircuit length'])
        
        #writer.writeheader()
        writer.writerow({'filename': IMAGE_FILENAME.split(sep='/')[2],
                        'initial terms': old_num_toffolis,
                        'minimized terms': num_toffolis,
                        'ratio': 1 - (num_toffolis/old_num_toffolis),
                        'time taken': total_time,
                        'qcircuit length': len(qcircuit)})

# run on Qiskit
if RUN_QISKIT:
    seed = 0
    job = execute(qcircuit, backend=BACKEND)
    result = job.result()
    print(result, result.get_counts(qcircuit))

    