# https://github.com/yajurahuja/QuantumCGT/blob/242ba3d037f813359baa9ca60bf8ec958ef56400/tictactoe/Game.py
from graphics import *
from qiskit import *
from sys import*

class game:
	def __init__(self, size, winp, turn, boxsize):
		self.boardsize = size
		self.win_parameter = winp
		self.turn = 1
		self.game_circuit = QuantumCircuit(self.boardsize ** 2, self.boardsize ** 2) # nXn size quantum circuit
		self.simulator = Aer.get_backend('qasm_simulator')
		self.boxsize = boxsize
		self.board = [[ 0 for i in range(size)] for j in range(size)] # board matrix
		self.turn = 1
		self.boxes = [[self.Rect(i*self.boxsize, j*self.boxsize) for i in range(self.boardsize)]for j in range(self.boardsize)]
		self.moves = 1		
		self.choice = ''
		self.points = []
		self.control_position = {}
		self.target_position = {}
		self.update = []
		self.boxinfo = self.boxinfo_g()
		self.current_win = None 
		self.turn_txt = None
		self.message = None
		self.colour = ''

	def Rect(self, x, y): # Create a rectanble box using diagonal points
		return Rectangle(Point(x, y + self.boxsize),Point(x + self.boxsize, y))

	def start_window_g(self):
		start_window = GraphWin("Welcome",600,600)
		start_window.setBackground('Black')
		qttt = Text(Point(300,50),"Quantum Tic-Tac-Toe")
		qttt.setSize(35)
		qttt.setTextColor(color_rgb(0,255,200))
		qttt.draw(start_window)
		welcome_msg ='Legal Moves:\n'\
		'- Classical Move: Collapses Box to black or white with equal probability\n'\
		'- Quantum Move: Entangles target box and control box. Reverses \ntarget box if control box collapses to black\n\n'\
		'Rules:\n'\
		'1) Initially all boxes are in Quantum State/superposition of \nblack and white\n'\
		'2) Only Classical move can collapses a quantum state to a \nclassical state\n'\
		'3) A  classical  move  can  only  be  applied  to  a  box  that\n  is in Quantum State '\
		'i.e. a classical  move cannot be apllied\n to the same box twice\n'\
		'4) The Quantum Move’s target box should be in a Classical \nState, and the control box should be in a Quantum State'
		start_window.setCoords(0,500,500,0)
		txt = Text(Point(250,200),welcome_msg)
		txt.setTextColor(color_rgb(0,255,200))
		txt.setSize(15)
		txt.draw(start_window)
		click = Rectangle(Point(200,375),Point(300,425))
		click.setFill('Orange')
		click.draw(start_window)
		click_txt = Text(Point(250,400),"Start")
		click_txt.setSize(20)
		click_txt.draw(start_window)
		while True:
			click_cord = start_window.getMouse()
			x = click_cord.getX()
			y = click_cord.getY()
			if 200<=x<=300 and 375<=y<=425:
				start_window.close()
				break


	def boxes_g(self, window):
		for row in self.boxes:
			for box in row:
				box.draw(window) 
				box.setFill("Grey")

	def main_window_g(self):
		main_win = GraphWin("Quantum Tic Tac Toe ",self.boardsize * self.boxsize, self.boardsize * self.boxsize+50)
		main_win.setCoords(0, self.boardsize * self.boxsize+50, self.boardsize * self.boxsize,0)
		self.current_win = main_win
		self.boxes_g(main_win)


	def setup(self): # Setup the game circuit and initial display screen
		for i in range(self.boardsize ** 2):
			self.game_circuit.h(i)
		self.start_window_g() #Draw the window

	def control_target_g(self, window): 
		for index in range(self.boardsize ** 2):
			row = index//self.boardsize
			column = index%self.boardsize
			print(row, column)
			if index+1 in self.control_position.keys():
				self.boxinfo[row][column].undraw()
				self.boxinfo[row][column].setText(str(control_pos[i+1]))
				self.boxinfo[row][column].draw(window)
			elif index+1 in self.target_position.keys():
				self.boxinfo[row][column].undraw()
				self.boxinfo[row][column].setText(str(target_pos[i+1]))
				self.boxinfo[row][column].draw(window)
			else:
				self.boxinfo[row][column].undraw()
				self.boxinfo[row][column].setText(str(list()))
				self.boxinfo[row][column].draw(window)

	def boxinfo_g(self):
		B = []
		for row in range(self.boardsize):
			T = []
			for column in range(self.boardsize):
				txt = Text(Point(column*self.boxsize+50,row*self.boxsize+10),'')
				txt.setTextColor("Green")
				T.append(txt)
			B.append(T)
		print(len(B))
		return B


	def play(self):

		self.control_target_g(self.current_win)
		self.turn_txt = Text(Point(self.boxsize*(self.boardsize/2), self.boxsize*self.boardsize+15),'')
		self.turn_txt.setSize(15)
		self.message = Text(Point(self.boxsize*(self.boardsize/2), self.boxsize*self.boardsize+35), '')
		self.message.setSize(15)
		self.message.draw(self.current_win)
		self.turn_txt.draw(self.current_win)

		while True:
			if self.turn == 1:
				self.colour = "White"
			else:
				self.colour = "Black"
			x, y = self.play_g()
			if(x == None or y == None):
				continue
			else:
				ch = self.choice_g()
				print(self.turn, self.moves, ch)
				self.turn = 3- self.turn
				self.moves = self.moves + 1



	def play_g(self):
		self.turn_txt.undraw()
		self.turn_txt.setText('Turn : Player '+str(self.turn)+'('+self.colour+')')
		self.turn_txt.draw(self.current_win)
		curr = self.current_win.getMouse()
		y = int(curr.getX() / self.boxsize) 
		x = int(curr.getY() / self.boxsize) 
		if not (0<=x<self.boardsize and 0<=y<self.boardsize):
			return None, None
		else:
			return x, y



	def choice_g(self):
		bsize = 100
		s = 2
		rec = [[self.Rect(i*bsize,50)] for i in range(s)]
		hchoice = self.Rect(50,150)
		win_choice = GraphWin("Choose Move",(s) * bsize,(s) * bsize+100)
		win_choice.setCoords(0,(s) * bsize+100,(s) * bsize,0)
		for x in rec: 
			for y in x:
				y.draw(win_choice)
		hchoice.draw(win_choice)

		txt = Text(Point(50,100),"Q move")
		txt.draw(win_choice)
		txt = Text(Point(150,100),"C move")
		txt.draw(win_choice)
		txt = Text(Point(100,200),"H move")
		txt.draw(win_choice)

		f=1
		while f == 1:
			p = win_choice.getMouse()
			y = int(p.getX() /self.boxsize)
			x = p.getY()-50
			xx = int(x /self.boxsize)
			
			if y==0 and xx==0 and x>=0:
				win_choice.close()
				return 'q'
			        
			elif y==1 and xx==0 and x>=0:
				win_choice.close()
				return 'c'

			elif 50<= p.getX() <=150 and 150<= p.getY() <=250:
				win_choice.close()
				return 'h'



	def cmove(self, row, column):
		index = row * self.boardsize + (column + 1)
		self.game_circuit.measure(index-1, self.boardsize**2 - index) 
		result = execute(self.game_circuit, get_backend, shots = 1).result()
		circuit_array = list(result.get_counts(circuit).keys())[0]
		error = 0
		if index not in self.points:
			if circuit_array[index-1] == '1':
				self.game_circuit.reset(index-1)
				self.game_circuit.x(index-1)
				circuit.measure(pos-1, self.boardsize**2 - index)
			else:
				self.game_circuit.reset(index-1)
				self.game_circuit.measure(index-1, self.boardsize**2 - index)

			#check the elements which are control/target points

		else:
			global message
			message.setText('Illegal move')
			message.draw(self.current_win)
			error=1
		points.append(index) #update the points to show the index has been measured
		self.colour_g()
		return error

	def hmove(self, row, column):
		index  = (row * self.boardsize) + (column + 1)
		if index in points and index not in self.target_position.keys():
			self.game_circuit.reset(index - 1)
			self.game_circuit.h(index - 1)
			self.boxes[row][column].setFill('Gray')
			self.points.remove(index)
			self.board[row][column] = 0
			return 0
		else: 
			global message
			message.setText('Illegal move')
			message.draw(win)
			return 1

	#def colour_g(self):
	
	def choice(self):
		return self.choice_g() #replace by strategy function

#	def move(self):


"""

	def boolp(x,y,n):
	if x >= size or y>=size or x<0 or y<0: 
		return False
	return n == board[x][y]

	def check_winner(): #Check if a  player is the winner
		for row in range(boardsize):
			for column in range(boardsize):
				box = board[row][column]
				if box == 0: continue

				#Check row
				check = True
				for i in range(winp):
					check = probe(row , column+i, box)
					if not check: break
					else: return box

				#Check column
				check = True
				for i in range(winp):
					check = boolp(row + i, column, box)
					if not check: break
					else: return box 
				
				#Check diagonal1
				check = True
				for i in range(winp):
					check = boolp(row + i, column + i, box)
					if not check: break
					else: return box

				#Check diagonal2
				check = True
				for i in range(winp):
					check = boolp(row - i , column + i, box)
					if not check: break
					else: return box
	return 0

	def interactive_play():
		col = ''
		while True: 
			if turn==1 : 
				col = 'White' 
			else:
				col = 'Black'
			turn_txt.undraw()
			turn_txt.setText('Turn : Player '+str(turn)+'('+col+')')
			turn_txt.draw(win)
			p = win.getMouse()
			y = int(p.getX() / boxSize) 
			x = int(p.getY() / boxSize) 
			if not (0<=x<size and 0<=y<size):
				continue
			ch = choice()
			message.undraw()
			if ch=='c':
				if l[x][y] > 0:
					message.setText('Illegal move')
					message.draw(win)
					continue
				coord = (x*size) + (y)
				err = cmove(x,y)
				modify_subscript()
				if err==1:
					continue
			elif ch=='q':
				if l[x][y] > 0:
					message.setText('Illegal move')
					message.draw(win)
					continue
				pp = win.getMouse()
				yy = int(pp.getX() / boxSize) 
				xx = int(pp.getY() / boxSize)
				coord_control = (x*size) + (y+1)
				coord_target = (xx*size)+ (yy+1)
				err = qmove(coord_control,coord_target)
				modify_subscript()
				if err==1:
					continue
			elif ch=='h':
				err = hmove(x,y)
				if err==1:
					continue

			turn = 3 - turn 
			status = check()
			if status == 1 : 
				print ("Player 1 wins")
				highlight("Dark Red")
				display_win(1)
				break
			if status == 2:
				print ("Player 2 wins")
				highlight("Dark Green")
				display_win(2)
				break
			moves += 1 

		if moves == size*size :
			print ("Draw match")
			for x in rec:
				for y in x:
					y.setFill("Blue")



	#should make a draw function for each of the graphic
	#move function
	#player class 
	# -player class will have different move functions 
	# we will call the move functions from inside the move function of the game class
	
"""


	