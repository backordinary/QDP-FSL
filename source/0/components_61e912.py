# https://github.com/LuisMi1245/QPath-and-Snakes/blob/0ad128221e462d3d8bb48624c14ad2f806e6731b/components.py
#Aquí van los componentes interactivos de la interfaz como botones, gráficas y demás
import pygame
from PIL import Image
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt    
import surfaces

def plot_qc(quantum_circuit):
    bk = Aer.get_backend('qasm_simulator') #backend
    job = execute(quantum_circuit, bk, shots=10000) 
    result = job.result()
    count = result.get_counts(quantum_circuit)
    plot_histogram(count)
    label = list(count.keys()) #x axis
    values = list(count.values())
    percentage = [round(i/sum(values),2) for i in values] #y axis
    x = np.arange(len(label))  # the label locations x
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, percentage, width)
    #ax.set_ylabel('Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(label, fontsize=30)
    ax.bar_label(rects1, padding=15, fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    #plt.title('Probabilities', fontsize=30)
    fig.tight_layout()
    plt.savefig("__stored_img__/plot_qcc.png")
    img = Image.open('__stored_img__/plot_qcc.png')
    new_img = img.resize((11*surfaces.bloque,11*surfaces.bloque))
    new_img.save('__stored_img__/plot_qcc.png','png')


# surface = superficie donde se ubica y se mide el botón (objeto surface)
# x,y = posicón (flotante)
# b,h = base y altura (flotante)
# btn_color = color del botón  (tupla)
# msg = mensaje del botón (texto)
# msg_size = tamaño del texto del botón (entero)
# msg_color = color del texto del botón (tupla)
# action = acción que genera el botón al ser presionado

def button(surface, x, y, b, h, btn_color, msg, msg_size, msg_color=(0,0,0), font_path="assets/fonts/Woodstamp.otf", action=None):
    mouse_x, mouse_y = pygame.mouse.get_pos() #obtiene la posición del puntero en la ventana. Tupla (x,y)
    #click = pygame.mouse.get_pressed() #captura el evento de presionar click
    
    abs_pos_surface_x, abs_pos_surface_y = pygame.Surface.get_abs_offset(surface) #Obtiene la posición absoluta de una "surface" con respecto a la surface de primer nivel 
    abs_pos_bttn_x, abs_pos_bttn_y = abs_pos_surface_x + x, abs_pos_surface_y + y #Obtiene la posición absoluta de un botón con respecto a la superficie de primer nivel

    if (abs_pos_bttn_x + b) > mouse_x > (abs_pos_bttn_x) and (abs_pos_bttn_y + h) > mouse_y > (abs_pos_bttn_y):
        #Si el puntero está sobre el botón, imprime forma del botón con hover
        pygame.draw.rect(surface, btn_color+np.array((30,30,30)), (x,y,b,h), border_radius=5)
        #Si presionó el click, ejecuta action()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and action is not None:
                action()
    else:
        #Si el puntero está fuera del área del botón, imprime forma del botón sin hover
        pygame.draw.rect(surface, btn_color, (x,y,b,h), border_radius=5)
    
    #texto del botón
    surface_msg, rect_msg = text_objects(msg, msg_size, font_path, msg_color) #Llama a la función que genera texto.
    rect_msg.center = ((x + (b / 2)), (y + (h / 2))) #Actualiza la posición del centro de la rejilla del texto en el centro de la surface donde se encuentre.

    #Se superpone la superficie "suface_msg" con "surface" 
    surface.blit(surface_msg, rect_msg)
 

#text = texto a renderizar (string)
#font = objeto fuente de pygame (objeto)
#color = color del texto (tupla RGB)

def text_objects(text, size, font_path, color=(0,0,0)):
    #función que retorna una tupla del texto y del rectángulo en donde se encierra el texto
    font = pygame.font.Font(font_path, size)
    textsurface = font.render(text, True, color)
    return textsurface, textsurface.get_rect()

#xi = posición inicial en x de la primera celda de la cuadrícula (float)
#yi = posición inicial en y de la primera celda de la cuadrícula (float)
#b  = ancho de la celda (float)
#h  = alto de la celda (float)
#dimension = Tupla que contiene la cantidad de filas y columnas (row, column)
#surface = superficie donde desea ubicarles
def cuadricula(xi, yi, b, h, dimension, surface, color, rand_color=False, borde=0):
    rows, columns = dimension[0], dimension[1]
    celdas = [[0 for col in range(0,columns)] for row in range(0,rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            if rand_color:
                celdas[i][j] = pygame.draw.rect(surface, color[i+j], (xi+b*j,yi+h*i,b,h), borde)
            else:
                celdas[i][j] = pygame.draw.rect(surface, color, (xi+b*j,yi+h*i,b,h), borde)
                
    return celdas
