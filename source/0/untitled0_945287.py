# https://github.com/jasper-zheng/msc-coding-2-jasper-zheng/blob/5ed54d5ee06fa0fbe725f0dc1da9f99989605465/week_05/python-challenges/untitled0.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:55:41 2022

@author: winter_camp
"""


def factoriser(n):
    
    v = [True for i in range(n+1)]

    for i in range(2,n):
        if v[i]:
            k = 2
            while i*k <=n:
                v[i*k] = 0
                k+=1
            
    for i in range(n,2,-1):
        if v[i]:
            result = i
            break
    
    return result





#%%

#!pip install qiskit
#!pip install matplotlib
#!pip install pylatexenc

#%%

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

#%%

Q_simulator=Aer.backends(name='qasm_simulator')[0]

q = QuantumRegister(1) # fill the circuit with two qubits 
c = ClassicalRegister(1) # fill the circuit with two classical bits

entangled = QuantumCircuit(q, c)

entangled.h(0)

entangled.measure(0,0)

#%%

entangled.draw(output='mpl') 

#%%

#executes measurements
job=execute(entangled, Q_simulator, seed_simulator = 95, shots = 2048)

hist=job.result().get_counts() 

#%%
plot_histogram(data=hist, title="Entangled State")


#%%


from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')  

#%%

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt

#%%

im = rgb2gray(imread('dct.png'))

imF = dct2(im)
 
im1 = idct2(imF)


#%%

from skimage.io import imsave

#%%

imsave('imF.png', imF)

#%%

imF2 = imread('imF.png')


#%%


im2 = idct2(imF2/255)

#%%


im3 = idct2(imF)

#%%


imFA = imF2/255


#%%

im3 = idct2(imFA)



#%%

imFs = imF

#%%
imFs[8:] = 0
imFs[:,8:] = 0

#%%

im4 = idct2(imFs)
plt.imshow(im4)

#%%

imFs = imF[:8,:8]

#%%
im4 = idct2(imFs)
plt.imshow(im4)

#%%

imF = imF*10+127

imF_int = imF.astype("uint8")

#%%
imsave('imF_int1.png', imF_int)


#%%
imF2 = imread('imF_int1.png').astype(int)

#%%

imF2 = (imF2-127)/10


#%%
imm = imF2/255
im4 = idct2(imF2)
plt.imshow(im4)


#%%

grey = rgb2gray(imread('grey.png'))


#%%

grey_dct = dct2(grey)



#%%

from scipy.io import wavfile



#%%

samplerate, data3 = wavfile.read('22_delay_01.wav')

square = np.array(data3).reshape((512,512))

plt.imshow(idct2(square),cmap='gray')



#%%

# Images are signals

# This is not a square wave, this is a square

#%%

from scipy.io import wavfile
import numpy as np

#%%

# use scipy to load the wav file
_, wav_data = wavfile.read('images_are_waves.wav')

print(wav_data.shape) # (262144,)

# look at the shape of the data, it is an 1-D array, 
# however, the hint indicates that the desired shape is a square
# therefore, we are going to reshape the data into a square matrix

# FYI, the screenshot shows the dimension of the square is (512 * 512).

wave_2d = np.array(wav_data).reshape((512,512))



#%%

from scipy.fftpack import idct

# implement 2D IDCT
def idct2d(input_matrix):
    return idct(idct(input_matrix.T, norm='ortho').T, norm='ortho')  

#%%

import matplotlib.pylab as plt

plt.imshow(idct2d(wave_2d),cmap='gray')



