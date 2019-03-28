import numpy as np
# n >= 3
# Array of [State,Value,Actions] for i in 1 to n*2
class Gridworld:
	def __init__(self,*args):
		n = args[0]
		std = args[1]
		self.n = n
		self.grid = []
		self.start_state = 1
		self.terminal_state = n*n
		for i in range(n*n):
			self.grid.append([i+1,(np.random.random(1)[0]-0.5)*2*std,self.transitions(i+1)])
            
	def classification(self,state):
		x = self.n
		cat = ""
		if state == 1:
			cat = "BLC" #Bottom left corner
		elif state == x:
			cat = "BRC" #Bottom right corner
		elif state == (x*x-x+1):
			cat = "TLC" #Top left corner
		elif state == (x*x):
			cat = "TRC" #Top right corner
		elif 1<state and state<x:
			cat = "BE" #Bottom edge
		elif (x*x-x+1)<state and state<(x*x):
			cat = "TE" #Top edge
		elif state%x == 0 and 1<state/x and state/x<x:
			cat = "RE" #Right edge
		elif (state-1)%x == 0 and 0<((state-1)/x) and ((state-1)/x)<(x-1):
			cat = "LE" #Left edge
		else:
			cat = "M" #Middle
		return cat
            
	def transitions(self,state):
		cat = self.classification(state)
		if cat == "M":
			actions = ["T","R","B","L"]
		elif cat == "LE":
			actions = ["T","R","B"]
		elif cat == "RE":
			actions = ["T","B","L"]
		elif cat == "TE":
			actions = ["R","B","L"]
		elif cat == "BE":
			actions = ["T","R","L"]
		elif cat == "BLC":
			actions = ["T","R"]
		elif cat == "BRC":
			actions = ["T","L"]
		elif cat == "TLC":
			actions = ["R","B"]
		elif cat == "TRC":
			actions = ["B","L"]
		return actions

	def world(self):
		arr1 = []
		arr2 = []
		for i in self.grid:
			arr1.append([i[0],i[2]])
			arr2.append(i[1])
		return [arr1,arr2]

	def new_state(self,state,action):
		if action == "T":
			return(state + self.n)
		elif action == "B":
			return(state - self.n)
		elif action == "R":
			return(state+1)
		elif action == "L":
			return(state-1)
