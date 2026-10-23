import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy.fft import dst
from scipy.linalg import lu_factor, lu_solve
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d

from matplotlib import font_manager
from matplotlib import rcParams

font_path = "times+simsun.ttf"
if os.path.exists(font_path):
	font_manager.fontManager.addfont(font_path)
	prop = font_manager.FontProperties(fname=font_path)
	rcParams['font.sans-serif'] = [prop.get_name()]
else:
	# 仓库不附带字体文件，缺少时回退到系统字体
	rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 13
rcParams['axes.unicode_minus'] = False

class NumericalSolutionForPossionEquation:

	def __init__(
		self, 
		lattice_function:callable, # 格点函数
		x_max:float=1.0, # x轴范围：[0,x_max]
		y_max:float=1.0, # y轴范围：[0,y_max]
		n:int=10, # n+1等分x轴与y轴
		):
		self.x_max = x_max
		self.y_max = y_max
		self.lattice_function = lattice_function
		self.n = n

		self.x_step = x_max/(n+1)
		self.y_step = y_max/(n+1)

		self.x_points = np.linspace(0, x_max, n+2)[1:-1]
		self.y_points = np.linspace(0, y_max, n+2)[1:-1]
		self.matrix_F = lattice_function(self.x_points[:, None], self.y_points)


	def _generate_matrix_AB(self):
		mat = np.zeros((self.n, self.n))
		np.fill_diagonal(mat, 2)
		np.fill_diagonal(mat[:-1, 1:], -1)
		np.fill_diagonal(mat[1:, :-1], -1)
		return mat

	def _generate_eigenvalues_AB(self, step):
		return 4/step**2*np.array([np.sin(i*np.pi*step/2)**2 for i in range(1, self.n+1)])

	def method_dst(self):
		""" 离散傅里叶正弦变换求解 """
		dst_transform = lambda mat: dst(dst(mat, type=1, axis=1),type=1, axis=0)/4
		eigenvalues_A = self._generate_eigenvalues_AB(self.x_step)
		eigenvalues_B = self._generate_eigenvalues_AB(self.y_step)
		V = dst_transform(-self.matrix_F) # 第一个大型矩阵运算：PFQ
		W = V/(eigenvalues_A[:, None] + eigenvalues_B)*4*self.x_step*self.y_step # 利用numpy的广播机制计算W
		U = dst_transform(W) # 第二个大型矩阵运算：PWQ
		return U

	def method_LUdecomposition(self):
		""" 将矩阵U拉直，转化为线性方程组使用LU分解求解 """
		matrix_A = 1/self.x_step**2*self._generate_matrix_AB()
		matrix_B = 1/self.y_step**2*self._generate_matrix_AB()
		I = np.eye(self.n)
		coefficient_matrix = np.kron(I, matrix_A) + np.kron(matrix_B, I)
		resultant_vector = -self.matrix_F.reshape(self.n*self.n, 1)

		lu, piv = lu_factor(coefficient_matrix)
		x = lu_solve((lu, piv), resultant_vector)
		U = x.reshape(self.n, self.n)
		return U

	def method_jacobi_iteration(self, max_iter=20000, tol=1e-4):
		""" 利用Jacobi迭代法求解 """
		iter_count = 0
		U_new = np.zeros((self.n, self.n))
		U_old = np.zeros((self.n, self.n))
		while iter_count < max_iter:
			U_old_pad = np.pad(U_old, pad_width=1, mode='constant', constant_values=0)
			for i in range(self.n):
				for j in range(self.n):
					i_pad = i+1; j_pad = j+1
					value_1 = (U_old_pad[i_pad+1, j_pad]+U_old_pad[i_pad-1, j_pad])/self.x_step**2
					value_2 = (U_old_pad[i_pad, j_pad+1]+U_old_pad[i_pad, j_pad-1])/self.y_step**2
					value_3 = self.matrix_F[i, j]
					value_4 = (2/self.x_step**2+2/self.y_step**2)
					U_new[i, j] = (value_1 + value_2 - value_3)/value_4

			diff = np.max(np.abs(U_old - U_new))
			if diff < tol:
				break
			iter_count += 1
			U_old = U_new.copy()
		else:
			print(f"注意：本方法(method_jacobi_iteration)已经超过最大迭代数{max_iter}，结果可能不收敛。")
		return U_new

	def method_jacobi_iteration_convolve(self, max_iter=20000, tol=1e-4):
		""" 利用Jacobi迭代法求解，迭代过程用scipy提供的卷积函数加速 """
		factor = 1 / (2/self.x_step**2 + 2/self.y_step**2)
		kernel = np.array([
			[0, 1/self.y_step**2, 0], 
			[1/self.x_step**2, 0, 1/self.x_step**2], 
			[0, 1/self.y_step**2, 0],
			]) * factor
		iter_count = 0

		U_new = np.zeros((self.n,self.n))
		U_old = np.zeros((self.n,self.n))

		while iter_count < max_iter:
			U_new = convolve2d(U_old, kernel, mode='same', boundary='fill', fillvalue=0) - self.matrix_F * factor
			diff = np.max(np.abs(U_old - U_new))
			if diff < tol:
				break
			U_old = U_new.copy()
			iter_count += 1
		else:
			print(f"注意：本方法(method_jacobi_iteration_convolve)已经超过最大迭代数{max_iter}，结果可能不收敛。")
		return U_new

	def method_gauss_seidel_iteration(self, max_iter=20000, tol=1e-4):
		""" 利用Gauss Seidel迭代法求解 """
		iter_count = 0
		U_new = np.zeros((self.n, self.n))
		U_old = np.zeros((self.n, self.n))

		while iter_count < max_iter:
			for i in range(self.n):
				for j in range(self.n):
					u_left = U_new[i-1, j] if i > 0 else 0
					u_right = U_old[i+1, j] if i < self.n-1 else 0
					u_down = U_new[i, j-1] if j > 0 else 0
					u_up = U_old[i, j+1] if j < self.n-1 else 0

					value_1 = (u_left + u_right) / self.x_step**2
					value_2 = (u_down + u_up) / self.y_step**2
					value_3 = self.matrix_F[i, j]
					value_4 = (2 / self.x_step**2 + 2 / self.y_step**2)

					U_new[i, j] = (value_1 + value_2 - value_3) / value_4

			diff = np.max(np.abs(U_new - U_old))
			if diff < tol:
				break

			U_old = U_new.copy()
			iter_count += 1
		else:
			print(f"注意：本方法(method_gauss_seidel_iteration)已经超过最大迭代数{max_iter}，结果可能不收敛。")

		return U_new

	def method_sor_iteration(self, omega=1.5, max_iter=20000, tol=1e-4):
		""" 利用SOR迭代法求解 """
		iter_count = 0
		U_new = np.zeros((self.n, self.n))
		U_old = U_new.copy()

		while iter_count < max_iter:
			U_old = U_new.copy()
			for i in range(self.n):
				for j in range(self.n):
					u_left = U_new[i-1, j] if i > 0 else 0
					u_right = U_old[i+1, j] if i < self.n-1 else 0
					u_down = U_new[i, j-1] if j > 0 else 0
					u_up = U_old[i, j+1] if j < self.n-1 else 0

					value_1 = (u_left + u_right) / self.x_step**2
					value_2 = (u_down + u_up) / self.y_step**2
					value_3 = self.matrix_F[i, j]
					value_4 = (2 / self.x_step**2 + 2 / self.y_step**2)

					U_new[i, j] += omega * (((value_1 + value_2 - value_3) / value_4) - U_new[i, j])

			diff = np.max(np.abs(U_new - U_old))
			if diff < tol:
				break

			iter_count += 1

		else:
			print(f"注意：本方法(method_sor_iteration)已经超过最大迭代数{max_iter}，结果可能不收敛。")

		return U_new

	def check_self(self, U):
		tol = np.mean(np.abs(self.matrix_F))/100
		kernel = np.array([
			[0, 1 / self.y_step**2, 0],
			[1 / self.x_step**2, -2 * (1 / self.x_step**2 + 1 / self.y_step**2), 1 / self.x_step**2],
			[0, 1 / self.y_step**2, 0],
			])

		F_estimated = convolve2d(U, kernel, mode='same', boundary='fill', fillvalue=0)
		diff = np.abs(F_estimated - self.matrix_F)
		max_diff = np.max(diff)
		mean_diff = np.mean(diff)

		return max_diff, mean_diff

	def check_analytical_solution(self, U, analytical_solution):
		X, Y = np.meshgrid(self.x_points, self.y_points, indexing='ij')
		U_analytical = analytical_solution(X, Y)
		max_diff = np.max(np.abs(U - U_analytical))
		mae = np.mean(np.abs(U - U_analytical))

		return max_diff, mae

	def visualize(self, U):
		X, Y = np.meshgrid([0]+list(self.x_points)+[self.x_max], [0]+list(self.y_points)+[self.y_max])
		U_pad = np.pad(U, pad_width=1, mode='constant', constant_values=0)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		surf = ax.plot_surface(X, Y, U_pad.T, cmap='Blues')
		fig.colorbar(surf, shrink=0.5, aspect=5)

		ax.set_title('Numerical Solution Visualize')
		ax.set_aspect('equal')
		plt.savefig("result.png", dpi=800)
		plt.show()

	def analyze_solution_time_error(self, analytical_solution=None):
		methods = [
			'method_dst', 
			'method_LUdecomposition', 
			'method_jacobi_iteration_convolve', 
			'method_sor_iteration', 
			'method_gauss_seidel_iteration', 
			'method_jacobi_iteration',  
			]
		times = []
		max_error = []
		mean_absolute_error = []
		print(f"初始化完成，准备开始求解微分方程。{self.n=}")
		for method in methods:
			start_time = time.time()
			U = getattr(self, method)()
			end_time = time.time()
			if analytical_solution:
				max_diff, mae = self.check_analytical_solution(U, analytical_solution)
				max_error.append(max_diff)
				mean_absolute_error.append(mae)
			times.append(end_time - start_time)
			print(f"\t{method}\t{end_time-start_time:.6f}秒。")
		
		solution_info = pd.DataFrame(times, index=[i.replace("method_","") for i in methods], columns=['Execution Time (s)'])
		if analytical_solution:
			solution_info["MAE"] = mean_absolute_error
			solution_info["Max Error"] = max_error
		return solution_info


def time_error_analysis(
	lattice_function, 
	x_max, y_max, 
	analytical_solution, 
	n_values, method_list=[
			'method_dst', 
			'method_LUdecomposition', 
			'method_jacobi_iteration_convolve', 
			'method_sor_iteration', 
			'method_gauss_seidel_iteration', 
			'method_jacobi_iteration',  
			]):
	output_folder = "Solution-Analysis"
	os.makedirs(output_folder, exist_ok=True)
	# 初始化存储结果的字典
	times = {method: [] for method in method_list}
	maes = {method: [] for method in method_list}
	max_errors = {method: [] for method in method_list}

	# 遍历不同的n值
	for n in n_values:
		print(n)
		solution = NumericalSolutionForPossionEquation(lattice_function, x_max, y_max, n)
		for method in method_list:
			start_time = time.time()
			U = getattr(solution, method)()
			end_time = time.time()

			max_diff, mae = solution.check_analytical_solution(U, analytical_solution)
			times[method].append(end_time - start_time)
			maes[method].append(mae)
			max_errors[method].append(max_diff)
	
	# 绘制求解时间与n的关系图像
	plt.figure(figsize=(10, 6))
	colors = plt.cm.viridis(np.linspace(0, 1, len(method_list)))
	for i, method in enumerate(method_list):
		plt.plot(n_values, times[method], label=method.replace("method_", "").replace("_", " "), color=colors[i], alpha=0.5)
	plt.xlabel('n')
	plt.ylabel('求解时间 (秒)')
	plt.title('求解时间与n的关系')
	plt.legend(loc='upper left')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
	plt.savefig(os.path.join(output_folder, "time_vs_n.png"), dpi=800)
	plt.close()

	# 绘制MAE和最大误差与n的关系图像
	fig, axs = plt.subplots(1, 2, figsize=(10, 6))
	for i, method in enumerate(method_list):
		axs[0].plot(n_values, maes[method], label=method.replace("method_", "").replace("_", " "), color=colors[i], alpha=0.5)
		axs[1].plot(n_values, max_errors[method], label=method.replace("method_", "").replace("_", " "), color=colors[i], alpha=0.5)
	
	axs[0].set_title('平均绝对误差(MAE)与n的关系')
	axs[1].set_title('最大误差与n的关系')
	for ax in axs:
		ax.set_xlabel('n')
		ax.set_ylabel('误差')
		ax.legend(loc='upper left')
		ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
	plt.savefig(os.path.join(output_folder, "errors_vs_n.png"), dpi=800)
	plt.show()
	plt.close()

	# 将数据保存到Excel文件中
	with pd.ExcelWriter(os.path.join(output_folder, "analysis_results.xlsx")) as writer:
		pd.DataFrame(times, index=n_values).to_excel(writer, sheet_name='Times')
		pd.DataFrame(maes, index=n_values).to_excel(writer, sheet_name='MAEs')
		pd.DataFrame(max_errors, index=n_values).to_excel(writer, sheet_name='Max Errors')


class NonhomogeneousPoissonEquation(NumericalSolutionForPossionEquation):
	def __init__(
		self, lattice_function: callable, 
		boundary_left: callable=lambda x: x*0, boundary_right: callable=lambda x: x*0, 
		boundary_top: callable=lambda x: x*0, boundary_bottom: callable=lambda x: x*0, 
		*args, **kwargs
	):
		super().__init__(lattice_function, *args, **kwargs)
		self.boundary_left = boundary_left
		self.boundary_right = boundary_right
		self.boundary_top = boundary_top
		self.boundary_bottom = boundary_bottom
		self._prepare_boundary_values()
		self.matrix_F_origin = self.matrix_F.copy()
		self._adjust_F_for_boundary()

	def _prepare_boundary_values(self):
		""" 根据边界条件计算非齐次方程 """
		x_points_total = np.linspace(0, self.x_max, self.n+2)
		y_points_total = np.linspace(0, self.y_max, self.n+2)
		self.boundary_values = np.zeros((self.n+2, self.n+2))
		self.boundary_values[:, 0] = self.boundary_left(y_points_total)
		self.boundary_values[:, -1] = self.boundary_right(y_points_total)
		self.boundary_values[0, :] = self.boundary_bottom(x_points_total)
		self.boundary_values[-1, :] = self.boundary_top(x_points_total)

	def _adjust_F_for_boundary(self):
		""" 根据非齐次边界条件调整F矩阵。调整之后不必重写LU分解与DST计算方法了。 """
		h2 = self.x_step**2
		k2 = self.y_step**2
		self.matrix_F[:, 0] -= self.boundary_values[1:-1, 0] / k2
		self.matrix_F[:, -1] -= self.boundary_values[1:-1, -1] / k2
		self.matrix_F[0, :] -= self.boundary_values[0, 1:-1] / h2
		self.matrix_F[-1, :] -= self.boundary_values[-1, 1:-1] / h2
		print(self.matrix_F)

	def method_jacobi_iteration_convolve(self, max_iter=20000, tol=1e-4):
		""" 卷积加速的Jacobi迭代法，适应非齐次边界条件 """
		factor = 1 / (2/self.x_step**2 + 2/self.y_step**2)
		kernel = np.array([
			[0, 1/self.y_step**2, 0], 
			[1/self.x_step**2, 0, 1/self.x_step**2], 
			[0, 1/self.y_step**2, 0],
			]) * factor
		iter_count = 0

		U_new = np.zeros((self.n,self.n))
		U_old = np.zeros((self.n,self.n))

		while iter_count < max_iter:
			# 应用边界条件
			U_with_boundary = np.pad(U_old, pad_width=1, mode='constant', constant_values=0)
			U_with_boundary[1:-1, 0] = self.boundary_values[1:-1, 0]    # Left boundary
			U_with_boundary[1:-1, -1] = self.boundary_values[1:-1, -1]  # Right boundary
			U_with_boundary[0, 1:-1] = self.boundary_values[0, 1:-1]    # Top boundary
			U_with_boundary[-1, 1:-1] = self.boundary_values[-1, 1:-1]  # Bottom boundary

			# 卷积运算
			U_new = convolve2d(U_with_boundary, kernel, mode='valid') - self.matrix_F_origin * factor

			diff = np.max(np.abs(U_old - U_new))
			if diff < tol:
				break
			U_old = U_new.copy()
			iter_count += 1
		else:
			print(f"注意：已经超过最大迭代数{max_iter}，结果可能不收敛。")
		return U_new

	def visualize(self, U):
		# 生成网格点，这次包括边界
		X, Y = np.meshgrid(
			[0] + list(self.x_points) + [self.x_max], 
			[0] + list(self.y_points) + [self.y_max]
		)

		# 为解U添加边界值
		U_with_boundary = np.pad(
			U, pad_width=1, mode='constant', constant_values=0
		)
		U_with_boundary += self.boundary_values
		
		# 创建3D图形
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		# 绘制曲面图
		surf = ax.plot_surface(X, Y, U_with_boundary.T, cmap='Blues')

		# 添加颜色条以表示高度值
		fig.colorbar(surf, shrink=0.5, aspect=5)

		# 设置标题
		ax.set_title('Numerical Solution with Nonhomogeneous Boundary Conditions')

		# 设置图形的长宽比为相等，这样x, y, z轴的一个单位将被显示为相同的长度
		ax.set_aspect('auto')

		# 保存图像
		plt.savefig("result_with_nonhomogeneous_boundary.png", dpi=800)

		# 显示图像
		plt.show()






def main():
	lattice_function = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
	analytical_solution = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
	time_error_analysis(lattice_function, 1, 1, analytical_solution, range(10,120,3))

	test = NumericalSolutionForPossionEquation(lattice_function,1.5,1.5)
	U = test.method_dst()
	test.visualize(U)

	def lattice_function(x, y):
		return 4 * np.ones_like(x) * np.ones_like(y)

	def boundary_left(y):
		return y**2

	def boundary_right(y):
		return 1 + y**2

	def boundary_bottom(x):
		return x**2

	def boundary_top(x):
		return x**2 + 1

	def analytical_solution(x, y):
		return x**2 + y**2
	test = NonhomogeneousPoissonEquation(lattice_function, boundary_left, boundary_right, boundary_top, boundary_bottom, n=40)
	U = test.method_dst()
	print(test.check_analytical_solution(U, analytical_solution))
	test.visualize(U)


if __name__ == "__main__":
	main()