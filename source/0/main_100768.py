# https://github.com/iQuHACK/2021_Shoe/blob/a8f73ddf5489cb64072770a44f11fa470d5bea18/main.py
import pygame
from pygame.locals import *
import os
import numpy as np
import sys
import math
import random
from qiskit import QuantumCircuit, execute, Aer

# pygame.init

qc = QuantumCircuit(2)
backend = Aer.get_backend('statevector_simulator')


pygame.init()

myfont = pygame.font.Font('freesansbold.ttf', 32)

cnum = input("Pick a level number (from 1 to 3): ")
level = int(cnum) #very important line!!!


W, H = 1200, 600
win = pygame.display.set_mode((W,H))
pygame.display.set_caption('Side Scroller')

bg = pygame.image.load(os.path.join('images','hbg.png')).convert()
bg = pygame.transform.scale(bg, (W, H))
bgX = 0
bgX2 = bg.get_width()

bell_state = False # they do not start in a bell state

clock = pygame.time.Clock()

class player(object):
    run = [pygame.image.load(os.path.join('images', 'beaver' + str(x) + '.png')) for x in range(8,16)]
    # run = [pygame.transform.scale(pygame.image.load(os.path.join('images', 'duck1.png')), (64, 64)) for x in range(8,16)]
    jump = [pygame.image.load(os.path.join('images', 'beaver' + str(x) + '.png')) for x in range(1,8)]
    slide = [pygame.image.load(os.path.join('images', 'beaver' + 'S1.png')),pygame.image.load(os.path.join('images', 'beaver' + 'S2.png')),pygame.image.load(os.path.join('images', 'beaver' +'S2.png')),pygame.image.load(os.path.join('images', 'beaver' +'S2.png')), pygame.image.load(os.path.join('images', 'beaver' +'S2.png')),pygame.image.load(os.path.join('images', 'beaver' +'S2.png')), pygame.image.load(os.path.join('images', 'beaver' +'S2.png')), pygame.image.load(os.path.join('images', 'beaver' +'S2.png')), pygame.image.load(os.path.join('images', 'beaver' +'S3.png')), pygame.image.load(os.path.join('images', 'beaver' +'S4.png')), pygame.image.load(os.path.join('images', 'beaver' +'S5.png'))]
    # jumpList = [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4]
    jumpList = [4,3,2,1,0,-1,-2,-3,-4, 0, 0]
    # fall = pygame.image.load(os.path.join('images','0.png'))
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.jumping = False
        self.small_jumping = False
        self.sliding = False
        self.slideCount = 0
        self.jumpCount = 0
        self.runCount = 0
        self.slideUp = False

    def draw(self, win):

        if self.jumping:
            self.y -= self.jumpList[self.jumpCount//12] * 1.2
            win.blit(self.jump[self.jumpCount//18], (self.x,self.y))
            self.jumpCount += 1
            if self.jumpCount > 108:
                self.jumpCount = 0
                self.jumping = False
                self.runCount = 0
            self.hitbox = (self.x+ 4,self.y,self.width-24,self.height-10)

        elif self.small_jumping:
            self.y -= self.jumpList[self.jumpCount//9] * 1.2
            win.blit(self.jump[self.jumpCount//15], (self.x,self.y))
            self.jumpCount += 1
            if self.jumpCount > 90:
                self.jumpCount = 0
                self.small_jumping = False
                self.runCount = 0
            self.hitbox = (self.x+ 4,self.y,self.width-24,self.height-10)

        elif self.sliding or self.slideUp:
            if self.slideCount < 20:
                self.y += 1
                self.hitbox = (self.x+ 4,self.y,self.width-24,self.height-10) # NEW

            elif self.slideCount == 80:
                self.y -= 19
                self.sliding = False
                self.slideUp = True

            elif self.slideCount > 20 and self.slideCount < 80: # NEW
                self.hitbox = (self.x,self.y+3,self.width-8,self.height-35) # NEW

            if self.slideCount >= 110:
                self.slideCount = 0
                self.slideUp = False
                self.hitbox = (self.x+ 4,self.y,self.width-24,self.height-10)
                self.runCount = 0

            win.blit(self.slide[self.slideCount//10], (self.x,self.y))
            self.slideCount += 1
            
        else:
            if self.runCount > 42:
                self.runCount = 0
            win.blit(self.run[self.runCount//6], (self.x,self.y))
            self.runCount += 1
            self.hitbox = (self.x+ 4,self.y,self.width-24,self.height-13)

        # pygame.draw.rect(win, (255,0,0),self.hitbox, 2)

class saw(object):
    #becoming stump
    # rotate = [pygame.image.load(os.path.join('images', 'SAW0.PNG')),pygame.image.load(os.path.join('images', 'SAW1.PNG')),pygame.image.load(os.path.join('images', 'SAW2.PNG')),pygame.image.load(os.path.join('images', 'SAW3.PNG'))]
    stump = pygame.image.load(os.path.join('images', 'stump.PNG'))
    def __init__(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 1.4
        if y > 200:
            self.stump = pygame.image.load(os.path.join('images', 'rock.PNG'))

    def draw(self,win):
        self.hitbox = (self.x + 10, self.y + 5, self.width - 40, self.height - 40)  # Defines the accurate hitbox for our character 
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)
       
        win.blit(self.stump, (self.x,self.y))  # scales our image down to 64x64 before drawing

    def collide(self, rect):

        if abs(rect[0] - self.hitbox[0]) > (rect[2] + self.hitbox[2])/2:
            return False
        if abs(rect[1] - self.hitbox[1]) > (rect[3] + self.hitbox[3])/2:
            return False

        return True

class spike(saw):  # We are inheriting from saw
    img = pygame.image.load(os.path.join('images', 'wood_spike.png'))
    def draw(self,win):
        self.hitbox = (self.x, self.y, self.width-10, self.height+40)  # defines the hitbox
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)
        win.blit(pygame.transform.scale(self.img, (self.width, self.height)), (self.x,self.y))

class big_rock(saw):
    img = pygame.image.load(os.path.join('images', 'cat.png'))
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        if self.y < 200:
            self.img = pygame.image.load(os.path.join('images', 'big_tree.png'))
    def draw(self,win):
        self.hitbox = (self.x+10, self.y+15, self.width-40, self.height-60)  # defines the hitbox
        win.blit(pygame.transform.scale(self.img, (self.width, self.height)), (self.x,self.y))
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)


class powerup(saw): #base powerup will be the H gate
    img = pygame.image.load(os.path.join('images', 'hadamard.png'))

    def draw(self,win):
        self.hitbox = (self.x, self.y, self.width-10, self.height-10)  # defines the hitbox
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)
        win.blit(pygame.transform.scale(self.img, (self.width, self.height)), (self.x,self.y))

class cnot(powerup):
    img = pygame.image.load(os.path.join('images', 'cnot.png'))

    def draw(self,win):
        self.hitbox = (self.x, self.y, self.width-10, self.height-10)  # defines the hitbox
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)
        win.blit(pygame.transform.scale(self.img, (self.width, self.height)), (self.x,self.y))

class finish(saw):
    img = pygame.image.load(os.path.join('images', 'finish.png'))

    def draw(self,win):
        self.hitbox = (self.x, self.y, self.width+10, self.height+10)  # defines the hitbox
        # pygame.draw.rect(win, (255,0,0), self.hitbox, 2)
        win.blit(pygame.transform.scale(self.img, (self.width, self.height)), (self.x,self.y))

    def collide(self, rect):
        if rect[0] > self.x + self.width/2:
            return True
        return False

# pygame.time.set_timer(USEREVENT+1, 500) # Sets the timer for 0.5 seconds
# # This should go above the game loop

runner = player(200, 465, 64, 64)
runner2 = player(200, 190, 64, 64)
# This should go above our game loop

# uncomment the 2 lines if you want random
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# pygame.time.set_timer(USEREVENT+1, random.randrange(1000, 1500)) # Will trigger every 2 - 3.5 seconds
# pygame.time.set_timer(USEREVENT+2, random.randrange(2000, 2500)) # Will trigger every 2 - 3.5 seconds

# This should go above the game loop

obstacles = []
run = True
# Should go above game loop

def redrawWindow():

    global run

    global bell_state # want to modify the global variable bell_state

    win.blit(bg, (bgX, 0))  
    win.blit(bg, (bgX2, 0))
    runner.draw(win) # NEW
    runner2.draw(win)

    # Loops through all obstacles
    for obstacle in obstacles:

        if isinstance(obstacle, finish):
            obstacle.draw(win)
            if obstacle.collide(runner.hitbox):
                ltext = myfont.render('You Win!', True, (0,0,0), (255, 255, 255))
                ltextRect = ltext.get_rect()
                ltextRect.center = (W // 2, H // 2 - 150)
                win.blit(ltext, ltextRect)

                print('You Win!')
                run = False

        elif isinstance(obstacle, powerup):
            # print('powerup')
            obstacle.draw(win)
            if obstacle.collide(runner.hitbox):
                if isinstance(obstacle, cnot):
                    qc.cnot(1, 0)
                else: 
                    qc.h(1)
                obstacles.pop(obstacles.index(obstacle))

            if obstacle.collide(runner2.hitbox):
                if isinstance(obstacle, cnot):
                    qc.cnot(0, 1)
                else: 
                    qc.h(0)
                obstacles.pop(obstacles.index(obstacle))


        else:

            # global run 

            obstacle.draw(win)
            if obstacle.collide(runner.hitbox):
                # pygame.time.delay(100)
                print('You Lose')
                ltext = myfont.render('You Lose', True, (0,0,0), (255, 255, 255))
                ltextRect = ltext.get_rect()
                ltextRect.center = (W // 2, H//2 - 150)
                win.blit(ltext, ltextRect)

                run = False
                # pass
            if obstacle.collide(runner2.hitbox):
                print('You Lose')

                ltext = myfont.render('You Lose', True, (0,0,0), (255, 255, 255))
                ltextRect = ltext.get_rect()
                ltextRect.center = (W // 2, H//2 - 150)
                win.blit(ltext, ltextRect)

                run = False


                # pass
                # pygame.time.delay(100)
                # runner.falling = True

    # update the text
    state = execute(qc,backend).result().get_statevector()

    num = [None] * 4
    aval = round(state[0].real, 2)
    bval = round(state[0].imag, 2)
    # print(i, 'here', aval+bval*1j)
    num[0] = aval + bval * 1j
    aval = round(state[1].real, 2)
    bval = round(state[1].imag, 2)
    # print(i, 'here', aval+bval*1j)
    num[1] = aval + bval * 1j
    aval = round(state[2].real, 2)
    bval = round(state[2].imag, 2)
    # print(i, 'here', aval+bval*1j)
    num[2] = aval + bval * 1j
    aval = round(state[3].real, 2)
    bval = round(state[3].imag, 2)
    # print(i, 'here', aval+bval*1j)
    num[3] = aval + bval * 1j

    # print(abs(num[0]), abs(num[1]), abs(num[2]), abs(num[3]))

    if abs(num[0]) > 0 and abs(num[3]) > 0 and abs(num[1]) < 0.001 and abs(num[2]) < 0.001:
        bell_state = True
        # print('Should be True')
    else:
        bell_state = False

    text = myfont.render(str(num), True, (0,0,0), (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (W // 2, H // 2)
    win.blit(text, textRect)

    pygame.display.update() 



speed = 100 # frame per second

delta = 1.8 # speed at which everything moves

# Sizes: STUMP = (64, 64)
# Sizes: Spike = (40, 120)
level_1 = [] # list of obstacles
level_1.append(saw(W+10, 185, 64, 64))
level_1.append(saw(W+210, 460, 64, 64))
level_1.append(spike(W+400, 50, 40, 160))
level_1.append(spike(W+600, 320, 40, 160))
level_1.append(finish(W+900, 40, 64, 200))
level_1.append(finish(W+900, 315, 64, 200))

level_2 = []
level_2.append(powerup(W+10, 380, 64, 64))
level_2.append(cnot(W+210, 380, 64, 64))
level_2.append(big_rock(W+410, 120, 64, 128))
level_2.append(big_rock(W+410, 400, 64, 128))
level_2.append(finish(W+610, 40, 64, 200))
level_2.append(finish(W+610, 315, 64, 200))

level_3 = []
level_3.append(powerup(W+10, 380, 64, 64))
level_3.append(cnot(W+210, 380, 64, 64))
level_3.append(big_rock(W+410, 120, 64, 128))
level_3.append(big_rock(W+410, 400, 64, 128))
level_3.append(cnot(W+610, 380, 64, 64))
level_3.append(saw(W+810, 460, 64, 64))
level_3.append(spike(W+810, 50, 40, 160))
level_3.append(finish(W+1010, 40, 64, 200))
level_3.append(finish(W+1010, 315, 64, 200))

if level == 1:
    for obstacle in level_1:
        obstacles.append(obstacle)
if level == 2:
    for obstacle in level_2:
        obstacles.append(obstacle)
if level == 3:
    for obstacle in level_3:
        obstacles.append(obstacle)

while run:

    redrawWindow() 
    bgX -= delta
    bgX2 -= delta

    if bgX < bg.get_width() * -1:  
        bgX = bg.get_width()
    
    if bgX2 < bg.get_width() * -1:
        bgX2 = bg.get_width()


    for obstacle in obstacles: 
        obstacle.x -= delta
        if obstacle.x < obstacle.width * -1: # If our obstacle is off the screen we will remove it
            obstacles.pop(obstacles.index(obstacle))
    # This should go in the game loop

    for event in pygame.event.get():  
        if event.type == pygame.QUIT: 
            run = False    
            pygame.quit() 
            quit()

        if event.type == USEREVENT+1:
            r = random.randrange(0,6)
            # r = 1 # only want the spikes
            if r == 0:
                obstacles.append(saw(W+10, 460, 64, 64))
            elif r == 1:
                obstacles.append(spike(W+10, 320, 40, 160))
            if r == 2:
                obstacles.append(saw(W+10, 185, 64, 64))
            elif r == 3:
                obstacles.append(spike(W+10, 50, 40, 160))
            elif r == 4:
                obstacles.append(big_rock(W+10, 400, 64, 128))
            elif r == 5:
                obstacles.append(big_rock(W+10, 120, 64, 128))

        if event.type == USEREVENT+2:
            r = random.randrange(0,3)

            if r == 0:
                obstacles.append(powerup(W+10, 380, 64, 64))

            if r == 1:
                obstacles.append(cnot(W+10, 380, 64, 64))

            if r == 2:
                obstacles.append(powerup(W+10, 105, 64, 64))
            
            # This should go in the "for event in pygame.event.get():" loop

    # Should go inside the game loop
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]: # If user hits space or up arrow key
        if bell_state:
            if (not(runner.jumping)) and (not(runner2.jumping)) and (not(runner.small_jumping)) and (not(runner2.small_jumping)):
                runner.jumping = True
                runner2.jumping = True
        else:
            if not(runner.small_jumping) and not(runner.jumping):  # If we are not already jumping
                runner.small_jumping = True
        # if not(runner.jumping):  # If we are not already jumping
            # runner.jumping = True

    if keys[pygame.K_DOWN]:  # If user hits down arrow key
        if bell_state:
            if (not(runner.sliding)) and (not(runner2.sliding)):
                runner.sliding = True
                runner2.sliding = True
        else:
            if not(runner.sliding):  # If we are not already jumping
                runner.sliding = True

    if keys[pygame.K_w]: # If user hits space or up arrow key
        if bell_state:
            if (not(runner.jumping)) and (not(runner2.jumping)) and (not(runner.small_jumping)) and (not(runner2.small_jumping)):
                runner.jumping = True
                runner2.jumping = True
        else:
            if not(runner2.small_jumping) and not(runner2.jumping):  # If we are not already jumping
                runner2.small_jumping = True

    if keys[pygame.K_s]:  # If user hits down arrow key
        if bell_state:
            if (not(runner.sliding)) and (not(runner2.sliding)):
                runner.sliding = True
                runner2.sliding = True
        else:
            if not(runner2.sliding):  # If we are not already jumping
                runner2.sliding = True

    # Because we have a starter file this is all we have to do to move our character. 
    # The physics and math behind the movement has been coded for you.

    clock.tick(speed) 

pygame.time.delay(2000)