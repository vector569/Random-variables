import numpy as np
import grid
import math

def gibbs(array,temperature):
	array1 = []
	sum_ = 0
	for i in range(len(array)):
		array1.append(math.exp(array[i]/temperature))
	for i in range(len(array1)):
		sum_ += array1[i]
	for i in range(len(array1)):
		array1[i] /= sum_
	return(array1)

def action(state,V,temp=1):
	action_set = world[state][1]
	Q = []
	for i in action_set:
		Q.append(V[obj.new_state(state,i)])
	action = np.random.choice(len(action_set),1, p=gibbs(Q,temp))[0]
	return action_set[action]

# 3X3, range_reward/range_noise = 5
n = 3
obj = grid.Gridworld(n,5)
[world,values] = obj.world()

# TD[0]
# Declaring initial values and policy
V = []
for i in range(n*n):
	V.append(0)

n_episodes = 1000
all_trails = []
alpha = 0.01
gamma = 0.8
temp = 1
for u in range(n_episodes):
	s = obj.start_state
	trail = [s]
	k = 1
	while(s != obj.terminal_state and k<1000):
		#A <- action from pi and s through softmax
		a = action(s,V,temp)
		[s_p,R] = obj.take_action(s,a)
		V[s] = V[s] + alpha*(1-u/n_episodes)*(R + gamma*(1-u/n_episodes)*V[s_p] - V[s])
		s = s_p
		trail.append(s)
		k += 1
	all_trails.append(trail)
	print(u+1)
print(all_trails[n_episodes-1])
print(V)