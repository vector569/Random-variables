# Metropolis-Hastings Algorithm

import numpy as np
import matplotlib.pyplot as plt
import math

class distributions:
	def __init__(self):
		pass
	dists = ['beta','logit_normal','uniform','irwin-hall','bates','kumaraswamy','pert',
                 'reciprocal','raised_cosine','triangular','truncated_normal','u_quadratic','wigner_semicircle',
                 'wrapped_exponential','wrapped_cauchy','beta_prime','birnbaum_sanders','logistic',
                 'champernowne','type-1_gumbel','generalized_normal','generalized_logistic','gumbel',
                 'hyperbolic','hyperbolic_secant','johnsons_su','laplace','von-mises',
                 'wrapped_asymptotic_laplace','chi','chi_square','inverse_chi_square','dagum','exponential',
                 'exponential_logarithmic','f_distribution','normal','frechet','erlang','generalized_pareto',
                 'gompertz','cauchy','lomax','nakagami','rayleigh','rice','weibull','gamma','pareto',
                 'students_t','fischers_z','generalized_extreme_value']
	
	def laplace(self,x,u=0,b=1):
		# x in R
		temp = (np.exp(-(abs(x-u))/b))/(2*b)
		return temp

	def generalized_logistic(self,x,alpha = 1):
		# alpha > 0 
		temp = (alpha*np.exp(-x))/((1+np.exp(-x))**(alpha+1))
		return temp

	def cauchy(self,x,alpha=1,u=0):
		temp = (alpha*alpha)/((alpha*np.pi)*((x-u)**2+alpha*alpha))
		return temp

	def hyperbolic_secant(self,x):
		temp = 0.5/(np.cosh(np.pi*x*0.5))
		return temp

	def normal(self,x,alpha=0,beta=1):
		temp = np.exp((-((x-alpha)**2))/(2*beta*beta))/((2*np.pi*beta*beta)**0.5)
		return temp

	def logistic(self,x,alpha=0,beta=1):
		temp = 1/(4*beta*((np.cosh((x-alpha)/(2*beta)))**2))
		return temp

	# def rayleigh(self,x,alpha=1):
	# 	# x > 0
	# 	temp = x*np.exp((-x*x)/(2*alpha*alpha))/(alpha*alpha)
	# 	return temp 
	# def beta(self,x,alpha=1.5,beta=1.5):
	# 	# 0 < x < 1
	# 	temp = ((x**(alpha-1))*((1-x)**(beta-1)))/((math.gamma(alpha)*math.gamma(beta))/(math.gamma(alpha+beta)))
	# 	return temp

	def plot(self,x,y,xlabel,ylabel):
		plt.plot(x,y)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
class metro_hastings:
	def q(self,x):
		y = np.random.normal(0,1,1)[0]
		return y

	def alpha(self,x,y,distr,obj):
		#print(x,y,distr)
		temp1 = getattr(obj,distr)(x)*self.q(x)
		if temp1>0:
			temp2 = getattr(obj,distr)(y)*self.q(y)
			temp2 = min(temp2/temp1,1)
		else:
			temp2 = 1
		return temp2

	def mha(self,x_j,distr,obj):
		u = np.random.uniform(0,1,1)[0]
		y = self.q(x_j)
		alp = self.alpha(x_j,y,distr,obj)
		if u<=alp:
			return y
		return x_j

	def simulate(self,n_iter,distr,obj,resolution,domain_min,domain_max,x0):
		x_p = np.arange(domain_min,domain_max,resolution)
		N = len(x_p)
		y_freq = [0]*N

		for k in range(n_iter):
			arr = []
			y_p = []
			arr = [x0]
			for i in range(N):
				y_p.append(getattr(obj,distr)(x_p[i]))
				x_a = self.mha(arr[-1],distr,obj)
				arr.append(x_a)

			for i in range(N):
				val = arr[i]
				j = int((val - domain_min)/resolution)
				j %= N
				y_freq[j] += 1/(N*resolution*n_iter)
			if k%(n_iter/10) == 0:
				print(str((k/(n_iter/10)+1)*10) + '% completed')
		return x_p,y_p,y_freq

n_iter = 500
distr = 'generalized_logistic'
obj1 = distributions()
obj2 = metro_hastings()
resolution = 0.01
domain_min = -10
domain_max = 10
x0 = 0
settings = {
	'generalized_logistic':[500,'generalized_logistic',obj1,0.01,-10,10,0],
	'laplace':[500,'laplace',obj1,0.01,-10,10,0], #Perfect
	'cauchy':[500,'cauchy',obj1,0.01,-10,10,0],
	'hyperbolic_secant':[500,'hyperbolic_secant',obj1,0.01,-10,10,0],  #Perfect
	'logistic':[500,'logistic',obj1,0.01,-10,10,0],

}

#sim_input = settings[distr]
#x_p,y_p,y_freq = obj2.simulate(sim_input[0],sim_input[1],sim_input[2],sim_input[3],sim_input[4],sim_input[5],sim_input[6])
x_p,y_p,y_freq = obj2.simulate(n_iter,distr,obj1,resolution,domain_min,domain_max,x0)
plt.plot(x_p,y_p,x_p,y_freq)
plt.show()
#obj1.plot(x_p,y_p,'X','p(X=x) Actual')
#obj1.plot(x_p,y_freq,'X','p(X=x) Simulated')
