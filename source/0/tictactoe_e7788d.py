# https://github.com/Janizai/Quantum-TicTacToe/blob/4eff646a75cc8b16024880f7d03afa3cf4ef822e/tictactoe.py
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)

class QuantumTicTacToe():
    def __init__(self) -> None: #Initialize all values in the class
        self.simulator = Aer.get_backend('statevector_simulator') #Initialize the simulator
        self.circuit = QuantumCircuit(9, 9) #Initialize the circuit
        
        self.status = np.zeros(9) #Initialize status to 0, 0 = no qubit, 1 = qubit exists, 3 = qubit measured
        self.board = np.zeros(9)  #Initialize board, which will be filled with the players' numbers
        
        self.player = 1 #Player 1 starts
        self.move_counter = 0 #Initialize move counter
        
        self.turn = 1 #Turn counter set to 1
        self.max_turns = 10 #Maximum number of turns reached

        self.win = False
        self.draw = False

    def update_status(self, playermove, reg0) -> None: #Update the status of the qubits
        if playermove == 0: #Initializing a new qubit
            self.status[reg0] = 1
        elif playermove == 3: #Measuring an existing qubit
            self.status[reg0] = 3

    def move(self, playermove, reg0): #Implement all possible moves on the board
        if playermove == 0: #Initializing a new qubit
            if self.player == 1: #Player 1
                self.circuit.id(reg0)
            
            if self.player == 2: #Player 2
                self.circuit.x(reg0)
                self.circuit.id(reg0)

        elif playermove == 1: #Move 1 = Hadamard gate
            self.circuit.h(reg0)
        
        elif playermove == 2: #Move 2 = X gate
            self.circuit.x(reg0)
        
        elif playermove == 3: #Move 3 = Measurement
            self.circuit.measure(reg0, reg0)
        
        #Update status
        self.update_status(playermove, reg0)
        
        #Simulate the circuit and get the resulting state vector
        job = execute(self.circuit, self.simulator)
        result = job.result()
        output_state = result.get_statevector()

        return output_state

    def is_valid_move(self, playermove, reg0): #Determine if a given move is valid
        message = ''
        if playermove == 0: #Initializing a new qubit
            if self.status[reg0] == 0:
                self.move_counter += 1
                self.update_status(playermove, reg0)
                status = True
            else:
                status = False
                message = 'A qubit has been allocated to the box.'
        
        elif playermove == 1 or playermove == 2: #H or X gate
            if self.status[reg0] == 1:
                self.move_counter += 1
                status = True
            
            if self.status[reg0] == 0:
                status = False
                message = 'There is no qubit in the box.'
            
            if self.status[reg0] == 3:
                status = False
                message = 'The qubit has been measured.'
        
        elif playermove == 3: #Measurement
            if self.status[reg0] == 1 and self.move_counter == 0:
                self.move_counter += 2
                self.update_status(playermove, reg0)
                status = True
            
            elif self.move_counter == 1:
                status = False
                message = 'You cannot perform a measurement after a unitary operation.'
            
            elif self.status[reg0] == 0:
                status = False
                message = 'There is no qubit in the box.'

            elif self.status[reg0] == 3:
                status = False
                message = 'The qubit has been measured already.'
        
        elif playermove == 4: #Skip a move or whole turn
            self.move_counter = 2
            status = True
        
        else: #Invalid move
            status = False
            message = 'Undefined operation.'
        
        return status, message

    def measurement_result(self, output_state, measured_register, qubitnumber): #Measure a qubit
        for index, element in enumerate(output_state):
            if element != 0:
                ket = bin(index)[2:].zfill(qubitnumber)
                result = ket[qubitnumber - measured_register - 1] #the ket is read from right to left(|987654321>)
                return result

    def turn_counter(self) -> None: #Turn counter
        if self.move_counter == 2 and not self.win and not self.draw: #2 moves and the game hasn't ended
            self.move_counter = 0
            self.turn += 1
            self.player = 2 - (self.player + 1) % 2 #Change player
    
    def check_for_win(self) -> bool: #Check if win condition has been fulfilled
        b = self.board
        if ((b[0] == 1 and b[1] == 1 and b[2] == 1) or # across the top
        (b[3] == 1 and b[4] == 1 and b[5] == 1) or # across the middle
        (b[6] == 1 and b[7] == 1 and b[8] == 1) or # across the bottom
        (b[0] == 1 and b[3] == 1 and b[6] == 1) or # down the left side
        (b[1] == 1 and b[4] == 1 and b[7] == 1) or # down the middle
        (b[2] == 1 and b[5] == 1 and b[8] == 1) or # down the right side
        (b[0] == 1 and b[4] == 1 and b[8] == 1) or # diagonal
        (b[2] == 1 and b[4] == 1 and b[6] == 1)):  # diagonal
            self.player = 1
            return True
        if ((b[0] == 2 and b[1] == 2 and b[2] == 2) or # across the top
        (b[3] == 2 and b[4] == 2 and b[5] == 2) or # across the middle
        (b[6] == 2 and b[7] == 2 and b[8] == 2) or # across the bottom
        (b[0] == 2 and b[3] == 2 and b[6] == 2) or # down the left side
        (b[1] == 2 and b[4] == 2 and b[7] == 2) or # down the middle
        (b[2] == 2 and b[5] == 2 and b[8] == 2) or # down the right side
        (b[0] == 2 and b[4] == 2 and b[8] == 2) or # diagonal
        (b[2] == 2 and b[4] == 2 and b[6] == 2)):  # diagonal
            self.player = 2
            return True
        return False

    def check_for_draw(self) -> bool: #Check if the game has ended in a draw
        count = np.count_nonzero(self.status == 3)
        
        if count == 9:
            return True
        else:
            return False

    def collapse_all(self) -> None: #Collapse all qubits once a certain number of turns has been reached
        for register, item in enumerate(self.status):
            if item == 1: #Measure all qubits that haven't been measured already
                output_state = self.move(3, register)
                res = self.measurement_result(output_state, register, 9)
                    
                if str(res) == '0':
                    self.board[register] = 1 #Player 1
                else:
                    self.board[register] = 2 #Player 2
            
        self.win = self.check_for_win()

        if not self.win:
            self.draw = self.check_for_draw()

    def step(self, action, reg0): #Execute a stat, out  in the game
        #Collapse the board
        if action == None and reg0 == None:
            self.collapse_all()
            stat = 2
            print(self.circuit)
            return stat, self.board

        #Check if the player is performing a valid move
        status, message = self.is_valid_move(action, reg0)
        
        stat = 1
        if not status and not self.win and not self.draw:
            return stat, message #Return error message
        
        #Valid move is executed
        output_state = self.move(action, reg0)

        out = None
        if action == 3: #Get measurement outcome
            res = self.measurement_result(output_state, reg0, 9)
            out = int(res)

            if str(res) == '0':
                self.board[reg0] = 1
            else:
                self.board[reg0] = 2

            self.win = self.check_for_win()
        
        print(self.circuit)

        if not self.win:
            self.draw = self.check_for_draw()
        
        self.turn_counter()

        stat = None
        return stat, out

    def reset(self) -> None: #Reset the game
        self.circuit = QuantumCircuit(9, 9)
        self.status = np.zeros(9)
        self.board = np.zeros(9)
        self.player = 1
        self.move_counter = 0
        self.turn = 0
        self.win = False
        self.draw = False

import time
import pygame as pg
from pygame.locals import *

class GUI():
    def __init__(self) -> None:
        self.width = 400  #Width of the window
        self.height = 400 #Height of the window
        self.fps = 60 #Set fps

        self.white = (255, 255, 255) #Background colour
        self.black = (0, 0, 0) #Line colour
        
        self.board = [[j for j in range(i, i + 3)] for i in range(0, 7, 3)] #Board indexing
        self.state = [[{'h' : 0, 'x' : 0} for j in range(i, i + 3)] for i in range(0, 7, 3)]
        
        self.player = 1 #Player
        self.turns = 1 #Turns
        self.move_count = 0 #Move count
        self.moves = {'q' : 0, 'h' : 1, 'x' : 2, 'm' : 3, 's' : 4} #Moves dictionary
        
        self.win = False #Win
        self.draw = False #Draw
        self.collapse = False #Collapse
        
        self.x_set = 100 #x offset
        self.y_set = 100 #y offset
        self.offset = (self.x_set, self.y_set)

        pg.init() #Initialize pygame window

        self.clock = pg.time.Clock() #Keep track of time
        
        self.screen = pg.display.set_mode(
            (self.width + 2 * self.x_set, self.height + 2 * self.y_set), 0, 32) #Screen
        
        pg.display.set_caption("Quantum Tic Tac Toe") #Set nametag

        #Load assets
        self.init_window = pg.image.load("./assets/cover.jpg")
        self.ket_x = pg.image.load("./assets/ketx.png")
        self.ket_0 = pg.image.load("./assets/ket0.png")
        self.qx = pg.image.load("./assets/x.png")
        self.q0 = pg.image.load("./assets/0.png")

        #Resize assets
        self.init_window = pg.transform.scale(self.init_window, 
        (self.width + 2 * self.x_set, self.height + 2 * self.y_set))
        self.ket_x = pg.transform.scale(self.ket_x, (80, 80))
        self.ket_0 = pg.transform.scale(self.ket_0, (80, 80))
        self.qx = pg.transform.scale(self.qx, (80, 80))
        self.q0 = pg.transform.scale(self.q0, (80, 80))

    def draw_status(self) -> None:
        #Marker
        if self.player == 1:
            marker = '0'
        else:
            marker = 'X'
        
        #Status message
        if not self.win:
            message = "Player {}'s turn. Marker: '{}'.".format(self.player, marker)
        
        else:
            message = "Player {} won!".format(self.player)
        
        if self.draw:
            message = 'Game ended in draw!'

        if not self.win and not self.draw and self.collapse:
            message = "Board state collapse in progress!"

        #Black edges
        self.screen.fill(self.black, (self.x_set, self.height + self.y_set, self.width, self.y_set))
        self.screen.fill(self.black, (self.x_set, 0, self.width, self.y_set))
        self.screen.fill(self.black, (0, 0, self.x_set, self.height + 2 * self.y_set))
        self.screen.fill(self.black, (self.width + self.x_set, 0, self.x_set, self.height + 2 * self.y_set))
        
        #Display information
        font = pg.font.Font(None, 30)

        text = font.render(message, 1, self.white)
        
        text_rect = text.get_rect(center = (self.width / 2 + self.x_set, self.height + 2 * self.y_set - 50))

        self.screen.blit(text, text_rect)

        text = font.render('Turn: {}   Move: {}'.format(self.turns, self.move_count), 1, self.white)
        
        text_rect = text.get_rect(center = (self.width / 2 + self.x_set, 50))
        
        self.screen.blit(text, text_rect)

        pg.display.update()

    def draw_grid(self) -> None:
        #Vertical lines
        pg.draw.line(self.screen, self.black, 
        (self.width / 3 + self.x_set, self.y_set), (self.width / 3 + self.x_set, self.height + self.y_set), 7)

        pg.draw.line(self.screen, self.black, 
        (self.width / 3 * 2 + self.x_set, self.y_set), (self.width / 3 * 2 + self.x_set, self.height + self.y_set), 7)

        #Horizontal lines
        pg.draw.line(self.screen, self.black, 
        (self.x_set, self.height / 3 + self.y_set), (self.width + self.x_set, self.height / 3 + self.y_set), 7)
        
        pg.draw.line(self.screen, self.black, 
        (self.x_set, self.height / 3 * 2 + self.y_set), (self.width + self.x_set, self.height / 3 * 2 + self.y_set), 7)

    def disp_init_window(self) -> None: #Display initial window
        self.screen.blit(self.init_window, (0, 0))

        pg.display.update()
        time.sleep(2)
        self.screen.fill(self.white)

        self.draw_grid()
        
        self.draw_status()

    def draw_ket(self, row, col) -> None: #Draw a ket
        posx = self.width / 3 * col + 30 + self.x_set
        posy = self.height / 3 * row + 30 + self.y_set

        if self.player == 1:
            self.screen.blit(self.ket_0, (posx, posy))
        else:
            self.screen.blit(self.ket_x, (posx, posy))

        font = pg.font.Font(None, 30)
        
        #The gate symbols
        text = font.render('{}:{}'.format('H', 0), 1, self.black)
        text_rect = text.get_rect(left = posx - 20, top = posy - 25, width = 30, height = 30)

        self.screen.blit(text, text_rect)

        text = font.render('{}:{}'.format('X', 0), 1, self.black)
        text_rect = text.get_rect(right = posx + 95, top = posy - 25, width = 30, height = 30)

        self.screen.blit(text, text_rect)

        pg.display.update()

    def draw_gate(self, row, col, gate) -> None: #Draw a gate
        posx = self.width / 3 * col + 30 + self.x_set
        posy = self.height / 3 * row + self.y_set

        #The gates are unitary: A^2 = I = A^0
        self.state[row][col][gate] = (self.state[row][col][gate] + 1) % 2

        font = pg.font.Font(None, 30)

        text = font.render('{}:{}'.format(gate.upper(), self.state[row][col][gate]), 1, self.black)
        
        if gate == 'h':
            text_rect = text.get_rect(left = posx - 20, top = posy + 5, width = 30, height = 30)
            self.screen.fill(self.white, (posx - 20, posy - 10, 40, 30))

        elif gate == 'x':
            text_rect = text.get_rect(right = posx + 95, top = posy + 5, width = 30, height = 30)
            self.screen.fill(self.white, (posx + 65, posy - 10, 30, 30))
        
        self.draw_grid()

        self.screen.blit(text, text_rect)

        pg.display.update()

    def draw_measurement(self, row, col, res) -> None: #Draw a classical bit
        posx = self.width / 3 * col + self.x_set
        posy = self.height / 3 * row + self.y_set

        self.screen.fill(self.white, (posx + 5, posy + 5, self.width / 3 - 10, self.height / 3 - 10))

        if res == 0:
            self.screen.blit(self.q0, (posx + 30, posy + 30))
        else:
            self.screen.blit(self.qx, (posx + 30, posy + 30))

        self.draw_grid()
        pg.display.update()

    def pos(self): #Get the mouse position on the grid
        x, y = pg.mouse.get_pos()

        col = None
        if (x - self.x_set < self.width / 3):
            col = 0
        elif (x - self.x_set < self.width / 3 * 2):
            col = 1
        elif (x - self.x_set < self.width):
            col = 2

        row = None
        if (y - self.y_set < self.height / 3):
            row = 0
        elif (y - self.y_set < self.height / 3 * 2):
            row = 1
        elif (y - self.y_set < self.height):
            row = 2
        
        return row, col

    def update(self, game) -> None: #Update game related values
        self.player = game.player
        self.move_count = game.move_counter
        self.turns = game.turn
        self.win = game.win
        self.draw = game.draw

    def keypress(self, game, key) -> None: #Handles key commands
        row, col = self.pos()

        if row != None and col != None:
            reg = self.board[row][col]
                        
            if key == K_q:
                stat, out = game.step(self.moves['q'], reg)
                            
                if stat == None:
                    self.draw_ket(row, col)

            elif key == K_h:
                stat, out = game.step(self.moves['h'], reg)
                            
                if stat == None:
                    self.draw_gate(row, col, gate = 'h')

            elif key == K_x:
                stat, out = game.step(self.moves['x'], reg)
                            
                if stat == None:
                    self.draw_gate(row, col, gate = 'x')
                        
            elif key == K_m:
                stat, out = game.step(self.moves['m'], reg)

                if stat  != None and type(out) == int:
                    self.draw_measurement(row, col, res = out)
                        
            elif key == K_SPACE or key == K_s:
                stat, out = game.step(self.moves['s'], reg)

            if stat == 1:
                print('Invalid move: {}'.format(out))

    def check_collapse(self, game) -> None: #Checks and performs board collapse
        if self.turns == game.max_turns:
            self.collapse = True
                
            self.draw_status()
                
            stat, out = game.step(None, None)
                
            for register, item in enumerate(out):
                row, col = np.where(np.asarray(self.board) == register)
                self.draw_measurement(row[0], col[0], item - 1)
                time.sleep(0.333333)
                
            self.update(game)
            self.draw_status()

    def run(self) -> None: #Runs the game
        game = QuantumTicTacToe()

        self.disp_init_window()

        running = True

        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

                    else:
                        self.keypress(game, event.key)

            self.update(game)
            self.draw_status()

            self.check_collapse(game)

            self.clock.tick(self.fps)

            if self.win or self.draw:
                running = False
                time.sleep(5)

        pg.quit()

if __name__ == '__main__':
    main = GUI()
    main.run()