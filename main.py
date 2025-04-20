import numpy as np
from scipy.optimize import linprog


class GameTheorySolver:
	def __init__(self, nature_probs, payoffs, know_a, know_b):
		"""
		Инициализация программы.
		"""
		self.p = np.array(nature_probs)
		self.W = payoffs
		self.know_a = know_a
		self.know_b = know_b
		self.strategies_a = ["11", "12", "21", "22"] if "nature" in know_a else ["1", "2"]
		self.strategies_b = ["11", "12", "21", "22"] if "nature" in know_b else ["1", "2"]
		self.rows = len(self.strategies_a)
		self.cols = len(self.strategies_b)

	def _build_state_matrix(self, state):
		"""
		Построение платёжной матрицы.
		"""
		matrix = np.zeros((self.rows, self.cols))
		knows_nature_a = "nature" in self.know_a
		knows_nature_b = "nature" in self.know_b
		knows_ab = "A" in self.know_b

		for i, a in enumerate(self.strategies_a):
			for j, b in enumerate(self.strategies_b):
				if knows_nature_a and knows_nature_b:
					key = f"{state}{a[state-1]}{b[state-1]}"
				elif knows_nature_a and not self.know_b:
					key = f"{state}{a[state-1]}{b}"
				elif knows_nature_a and knows_ab:
					key = f"{state}{a[state-1]}{b[int(a)-1]}"
				elif not self.know_a and not self.know_b:
					key = f"{state}{a}{b}"
				elif not self.know_a and knows_ab:
					key = f"{state}{a}{b[int(a)-1]}"
				elif not self.know_a and knows_nature_b:
					key = f"{state}{a}{b[state-1]}"
				else:
					raise ValueError("Неверная конфигурация")
				matrix[i, j] = self.W[key]
		return matrix

	def create_payoff_matrix(self):
		"""
		Построение итоговой матрицы.
		"""
		matrix1 = self._build_state_matrix(1)
		matrix2 = self._build_state_matrix(2)
		
		print("Матрица 1:\n", matrix1)
		print("Матрица 2:\n", matrix2)
		
		final_matrix = matrix1 * self.p[0] + matrix2 * self.p[1]
		
		print("Итоговая матрица:")
		for row in final_matrix:
			print("\t".join(f"{v:.2f}" for v in row))
				
		return final_matrix

	def remove_dominated_columns(self, matrix):
		"""
		Удаление доминирующих стратегий.
		"""
		cols, rows = matrix.shape[1], matrix.shape[0]
		keep = [True] * cols

		for j1 in range(cols):
			if not keep[j1]:
				continue

			for j2 in range(cols):
				if j1 == j2 or not keep[j2]:
					continue

				j1_dominates = all(matrix[i, j1] <= matrix[i, j2] for i in range(rows))
				strictly_better = any(matrix[i, j1] < matrix[i, j2] for i in range(rows))
				if j1_dominates and strictly_better:
					keep[j2] = False

		reduced_matrix = matrix[:, keep]
		return reduced_matrix

	def find_minimax(self, matrix):
		"""
		Нахождение минимакса.
		"""
		row_mins = np.min(matrix, axis=1)
		col_maxs = np.max(matrix, axis=0)
		maximin = np.max(row_mins)
		minimax = np.min(col_maxs)
			
		print(f"Максимин: {maximin:.2f}")
		print(f"Минимакс: {minimax:.2f}")
		
		return maximin, minimax

	def solve_game(self, matrix, maximin, minimax):
		"""
		Решение задачи.
		"""
		rows, cols = matrix.shape
		p_vector = np.zeros(rows)
		q_vector = np.zeros(cols)

		if np.isclose(maximin, minimax):
			row_idx = np.where(np.min(matrix, axis=1) == maximin)[0][0]
			col_idx = np.where(np.max(matrix, axis=0) == minimax)[0][0]
			p_vector[row_idx] = 1
			q_vector[col_idx] = 1
			game_value = maximin
			
			print(f"Найдена седловая точка: {game_value:.2f}")
			print(f"Цена игры: {game_value:.2f}")
			print(f"Вектор стратегий игрока A: [{', '.join(f'{v:.2f}' for v in p_vector)}]")
			print(f"Вектор стратегий игрока B: [{', '.join(f'{v:.2f}' for v in q_vector)}]")
		else:
			print("Седловой точки нет.")
			if matrix.shape == (2, 2):
				a11, a12 = matrix[0][0], matrix[0][1]
				a21, a22 = matrix[1][0], matrix[1][1]

				denom = a11 - a12 - a21 + a22

				p1 = (a22 - a21) / denom
				q1 = (a22 - a12) / denom
				
				p_vector = [p1, 1 - p1]
				q_vector = [q1, 1 - q1]

				game_value = (a11 * a22 - a12 * a21) / denom
				
				print(f"Цена игры: {game_value:.2f}")
				print(f"Вектор стратегий игрока A: [{', '.join(f'{v:.2f}' for v in p_vector)}]")
				print(f"Вектор стратегий игрока B: [{', '.join(f'{v:.2f}' for v in q_vector)}]")
			else:
				# Решение для игрока A
				c = np.zeros(rows + 1)
				c[-1] = -1

				A_ub = np.hstack([-matrix.T, np.ones((cols, 1))])
				b_ub = np.zeros(cols)

				A_eq = np.hstack([np.ones((1, rows)), np.zeros((1, 1))])
				b_eq = np.array([1.0])

				bounds = [(0, None)] * rows + [(None, None)]
				
				res_a = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
							
				if res_a.success:
					game_value = res_a.x[-1]
					p_vector = res_a.x[:-1]
					
					# Решение для игрока B
					c = np.zeros(cols + 1)
					c[-1] = 1

					A_ub = np.hstack([matrix, -np.ones((rows, 1))])
					b_ub = np.zeros(rows)

					A_eq = np.hstack([np.ones((1, cols)), np.zeros((1, 1))])
					b_eq = np.array([1.0])

					bounds = [(0, None)] * cols + [(None, None)]
					
					res_b = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
									
					if res_b.success:
						q_vector = res_b.x[:-1]

						print(f"Цена игры: {game_value:.2f}")
						print(f"Вектор стратегий игрока A: [{', '.join(f'{x:.4f}' for x in p_vector)}]")
						print(f"Вектор стратегий игрока B: [{', '.join(f'{x:.4f}' for x in q_vector)}]")
					else:
						print("Решение не найдено")
				else:
					print("Решение не найдено")

def main():
	"""Пример задачи."""
	
	# Вводные параметры
	p = [0.4, 0.6]
    
	W = {
		"111": -2, 
		"112": 4, 	
		"121": 1, 
		"122": -4,
		"211": 3, 
		"212": 0, 
		"221": -3, 
		"222": 5
	}
    
	know_a = []
	know_b = ["nature", "A"]

	solver = GameTheorySolver(p, W, know_a, know_b)
	
	matrix = solver.create_payoff_matrix()
	reduced_matrix = solver.remove_dominated_columns(matrix)
	maximin, minimax = solver.find_minimax(matrix)
	solver.solve_game(matrix, maximin, minimax)


if __name__ == "__main__":
	main()
