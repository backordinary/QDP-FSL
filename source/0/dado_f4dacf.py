# https://github.com/LuisMi1245/QPath-and-Snakes/blob/00a9139408b684fd44beadd06df7c3f703ec48c5/Dado.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import pygame

qc_dado = QuantumCircuit(3,3)
qc_dado.h(0)
qc_dado.h(1)
qc_dado.h(2)
qc_dado.measure((0,1,2),(0,1,2))

def dice_img(count):
    if count == '001':
        im = pygame.image.load('assets/img/img_dados/1.png')
    elif count == '010':
        im = pygame.image.load('assets/img/img_dados/2.png')
    elif count == '011':
        im = pygame.image.load('assets/img/img_dados/3.png')
    elif count == '100':
        im = pygame.image.load('assets/img/img_dados/4.png')
    elif count == '101':
        im = pygame.image.load('assets/img/img_dados/5.png')
    elif count == '110':
        im = pygame.image.load('assets/img/img_dados/6.png')
    return im 

def dice(qc):
    bk = Aer.get_backend('qasm_simulator')
    job = execute(qc, bk, shots=1)
    result = job.result()
    count = result.get_counts(qc)
    count = list(count.keys())[0]

    if (count == '000') or (count == '111'):
        count = dice(qc_dado)
        im = dice_img(count)
    else:
        im = dice_img(count)
    
    pygame.image.save(im, '__stored_img__/im_dado.png','png')
    return count