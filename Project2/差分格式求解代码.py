import numpy as np
import matplotlib.pyplot as plt

class ConvectionEquation:
	def __init__(self, a, h=0.1, tau=0.08):
		self.a = a
		self.h = h
		self.tau = tau
		self.lambda_ = self.tau / self.h
		self.x = np.arange(-10, 18, self.h)
		self.t = np.arange(0, 4 + self.tau, self.tau)
		self.u = np.full((len(self.t), len(self.x)), np.nan)

	def initial_condition(self):
		return np.array([1 if xi <= 0 else 0 for xi in self.x[1:-1]])

	def exact_solution(self):
		return np.array([1 if xi <= self.a * self.t[-1] else 0 for xi in self.x])

	def calculate_error(self):
		exact_u = np.array([1 if xi <= self.a * self.t[-1] else 0 for xi in self.x])
		epsilon = self.u[-1] - exact_u
		return np.sqrt(self.h * np.nansum(epsilon**2))

	def upwind_scheme(self):
		self.u[0, 1:-1] = self.initial_condition()
		for n in range(0, len(self.t) - 1):
			for j in range(1, len(self.x) - 1):
				self.u[n + 1, j] = self.u[n, j] - self.a * self.lambda_ * (self.u[n, j] - self.u[n, j - 1])

	def beam_warming_scheme(self):
		self.u[0, 1:-1] = self.initial_condition()
		for n in range(0, len(self.t) - 1):
			for j in range(2, len(self.x) - 1):
				self.u[n + 1, j] = (self.u[n, j] 
									- self.a * self.lambda_ * (self.u[n, j] - self.u[n, j - 1]) 
									- 0.5 * self.a * self.lambda_ * (1 - self.a * self.lambda_) * (self.u[n, j] - 2 * self.u[n, j - 1] + self.u[n, j - 2]))

	def lax_friedrichs_scheme(self):
		self.u[0, 1:-1] = self.initial_condition()
		for n in range(0, len(self.t) - 1):
			for j in range(1, len(self.x) - 1):
				self.u[n + 1, j] = 0.5 * (self.u[n, j + 1] + self.u[n, j - 1]) - 0.5 * self.a * self.lambda_ * (self.u[n, j + 1] - self.u[n, j - 1])

	def lax_wendroff_scheme(self):
		self.u[0, 1:-1] = self.initial_condition()
		for n in range(0, len(self.t) - 1):
			for j in range(1, len(self.x) - 1):
				self.u[n + 1, j] = (self.u[n, j] 
									- 0.5 * self.a * self.lambda_ * (self.u[n, j + 1] - self.u[n, j - 1]) 
									+ 0.5 * (self.a * self.lambda_)**2 * (self.u[n, j + 1] - 2 * self.u[n, j] + self.u[n, j - 1]))

	def plot_results(self, method_name):
		error = self.calculate_error()
		exact_u = self.exact_solution()
		
		valid_x = self.x[~np.isnan(self.u[-1])]
		valid_u = self.u[-1][~np.isnan(self.u[-1])]
		valid_exact_u = exact_u[~np.isnan(self.u[-1])]

		plt.figure(figsize=(10, 6))
		plt.plot(valid_x, valid_u, linestyle='--', color='#1f77b4', label=f'Numerical: {method_name}')
		plt.plot(valid_x, valid_exact_u, linestyle='-', color='#ff7f0e', alpha=0.7, label='Exact')
		plt.xlabel('x')
		plt.ylabel('u')
		plt.title(f'{method_name} (a={self.a}, t={self.t[-1]}) - L2 Error: {error:.4g}')
		plt.legend(loc='lower left')
		plt.grid(True)
		plt.xlim(valid_x[0], valid_x[-1])
		plt.savefig(f"{method_name}, a={self.a}.png", dpi=800)
		plt.show()

def main():
	speeds = [1, 2, 4]
	for speed in speeds:
		ce = ConvectionEquation(a=speed)
		ce.upwind_scheme()
		ce.plot_results("Upwind Scheme")

		ce = ConvectionEquation(a=speed)
		ce.beam_warming_scheme()
		ce.plot_results("Beam-Warming Scheme")

		ce = ConvectionEquation(a=speed)
		ce.lax_friedrichs_scheme()
		ce.plot_results("Lax-Friedrichs Scheme")

		ce = ConvectionEquation(a=speed)
		ce.lax_wendroff_scheme()
		ce.plot_results("Lax-Wendroff Scheme")


if __name__ == "__main__":
	main()