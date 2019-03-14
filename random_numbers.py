# Metropolis-Hastings Algorithm

import numpy as np
import matplotlib.pyplot as plt


class distributions:
	def __init__(self):
		pass
	dists = ['arcsine','beta','logit_normal','uniform','irwin-hall','bates','kumaraswamy','pert',
                 'reciprocal','raised_cosine','triangular','truncated_normal','u_quadratic','wigner_semicircle',
                 'wrapped_exponential','wrapped_cauchy','beta_prime','birnbaum_sanders','logistic',
                 'champernowne','type-1_gumbel','generalized_normal','generalized_logistic','gumbel',
                 'hyperbolic','hyperbolic_secant','johnsons_su','laplace','von-mises',
                 'wrapped_asymptotic_laplace','chi','chi_square','inverse_chi_square','dagum','exponential',
                 'exponential_logarithmic','f_distribution','normal','frechet','erlang','generalized_pareto',
                 'gompertz','cauchy','lomax','nakagami','rayleigh','rice','weibull','gamma','pareto',
                 'students_t','fischers_z','generalized_extreme_value']
	def test(self):
		print("Yep")
	def arcsine(self,x):
		# 0 < x <1
		temp = 1/(np.pi*(x*(1-x))**0.5)
		return temp 	
	
	def laplace(self,x,u=0,b=1):
		# x in R
		temp = (np.exp(-(abs(x-u))/b))/(2*b)
		return temp
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

	def simulate(self,n_iter,distr,obj,resolution,domain_min,domain_max):
		x_p = np.arange(domain_min,domain_max,resolution)
		N = len(x_p)
		y_freq = [0]*N

		for k in range(n_iter):
			x0 = 0
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
				if j>=N:
					j = N-1
				y_freq[j] += 1/(N*resolution*n_iter)
			if k%(n_iter/10) == 0:
				print(str((k/(n_iter/10)+1)*10) + ' \% completed')
		return x_p,y_p,y_freq

n_iter = 50
distr = 'laplace'
obj1 = distributions()
obj2 = metro_hastings()
resolution = 0.004
domain_min = -8
domain_max = 8

x_p,y_p,y_freq = obj2.simulate(n_iter,distr,obj1,resolution,domain_min,domain_max)

obj1.plot(x_p,y_p,'X','p(X=x) Actual')
obj1.plot(x_p,y_freq,'X','p(X=x) Simulated')
