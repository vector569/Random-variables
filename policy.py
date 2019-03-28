import numpy as np
import grid
# 3X3, range_reward/range_noise = 5
n = 3
obj = grid.Gridworld(n,5)
[world,values] = obj.world()

# Value iteration

# Declaring initial values and policy
pi = []
V = []
for i in range(n*n):
	V.append(0)
	pi.append(world[i][1][0])
print(pi)

epsilon = 0.005
delta = 0
while(delta<epsilon):
	for i in range(n*n):
		v = V[i]
		V[i] = max__
		delta = max(delta,abs(v-V[i]))


