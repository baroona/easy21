import numpy as np
import random
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker




class Player:
	def __init__(self):
		self.pv = None
		self.dv = None

	def playRandom(self, s, n):
		# playersum
		# dealer card
		reward = 0
		for x in range(1, n + 1):
			new_state = copy.deepcopy(s)
			while(new_state.isEnd() is False):
				if(np.random.random_sample() > 0.5):
					action = 1
				else:
					action = 0
				new_state = step(new_state, action)
			reward = reward + new_state.winner
			print("reward is", reward)
		return (reward / n)


class Play:
	def __init__(self):
		self.pv = None
		self.dv = None

	def playRandom(self, s, n):
		# playersum
		# dealer card
		reward = 0
		for x in range(1, n + 1):
			new_state = copy.deepcopy(s)
			while(new_state.isEnd() is False):
				if(np.random.random_sample() > 0.5):
					action = 1
				else:
					action = 0
				new_state = step(new_state, action)
			reward = reward + new_state.winner
			print("reward is", reward)
		return (reward / n)

	def naiveAction(self, state):
		if(state.pv < 19):
			return 1
		else:
			return 0

	def mcAction(self, state, V, N):
		N0 = 100
		e = 0
		#epsilon = N0/(N0 + )
		stick_reward = 0
		hit_reward = 0
		if(np.random.uniform() < e):  # choose randomly
			action = random.randint(0, 1)
		else:  # choose greedily with mc
			hit_reward = self.playRandom(step(copy.deepcopy(state), 1), 10)
			stick_reward = self.playRandom(step(copy.deepcopy(state), 0), 10)
			if(hit_reward >= stick_reward):
				action = 1
			else:
				action = 0
		return action


class Card:
	def __init__(self):
		self.v = np.random.randint(1, 11)

	def draw(self):
		if(np.random.random_sample() < (1 / 3)):
			self.v = -1 * np.random.randint(1, 11)
		else:
			self.v = 1 * np.random.randint(1, 11)

		return self.v


class State:
	def __init__(self):
		self.dv = Card().v
		self.pv = Card().v
		self.playerHasStuck = None
		self.dealerHasStuck = None
		self.end = None
		self.winner = 0  # reward

	def isEnd(self):
		# if self.end is not None:
			# print("first")
			# return self.end
		# print("player value is", self.pv)
		# print("dealer value is", self.dv)
		if(self.pv > 21 or self.pv < 1):
			# print("Player is bust")
			self.end = True
			self.winner = -1
			return self.end
		elif(self.dv > 21 or self.dv < 1):
			# print("Dealer is bust")
			self.end = True
			self.winner = 1
			return self.end
		elif((self.playerHasStuck is not None) and (self.dealerHasStuck is not None)):
			# print("Both have stuck")
			self.end = True
			if(self.pv > self.dv):
				self.winner = 1
			elif(self.pv < self.dv):
				self.winner = -1
			else:
				self.winner = 0
			return self.end
		self.end = False
		# print("game not finished")
		return self.end

	def playDealer(self):
		while (self.dv <= 17 and self.dv > 0):
			# print("dealer hits")
			self.dv = self.dv + Card().draw()
		# if(self.dv <= 17 & self.isEnd() is False)
# input: state s(dealer firstcard and player sum, action(hit(1)) or stick((0))
# output: sample s' of the next state and reward r


def step(s, action):
	if(action == 1):
		# print("player hits")
		s.pv = s.pv + Card().draw()
	else:
		s.playerHasStuck = True
		s.playDealer()
		s.dealerHasStuck = True
		# dealer finishes his game
	return s


# input:
	# n: number of games
	# ss




def playgames(n):
	res = np.zeros(n)
	for i in range(0, n):
		# print("game number", i)
		# create state
		s = State()
		p = Player(s)
		# print(s.isEnd())
		# print("player start value is", s.pv)
		# print("dealer start value is", s.dv)
		# k = 1
		# print("loop start")
		while(s.isEnd() is False):
			# print(k)
			action = p.naiveAction(s)
			step(s, action)
			# print("player value is", s.pv)
			# print("dealer value is", s.dv)
			# print("loophasrun")
			# k = k + 1
		# print("loopend")
		# print(s.isEnd())
		res[i] = s.winner
		# TODO
	return res



def recursiveMC(s, Q, N, N0):
	
	if(s.isEnd() is True):
		return s.winner

	else:
		epsilon = N0 / (N0 + (N[0, s.dv, s.pv] + N[1, s.dv, s.pv]))
		if(np.random.uniform() < epsilon):  # choose randomly
			action = random.randint(0, 1)
		else:
			action = np.argmax(Q[:, s.dv, s.pv])

		N[action, s.dv, s.pv] = N[action, s.dv, s.pv] + 1
		alpha = 1 / (N[action, s.dv, s.pv])
		g = recursiveMC(step(copy.deepcopy(s), action), Q, N, N0)
		# print("bef assignement", V[action, s.dv, s.pv])
		Q[action, s.dv, s.pv] = Q[action, s.dv, s.pv] + (alpha * (g - Q[action, s.dv, s.pv]))
		# print("after ass", V[action, s.dv, s.pv])
		return g


def MCgame(Q, N):
	s = State()
	N0 = 100
	path = []
	while(s.isEnd() is False):
		epsilon = N0 / (N0 + (N[0, s.dv, s.pv] + N[1, s.dv, s.pv]))
		if(np.random.uniform() < epsilon):  # choose randomly
			action = random.randint(0, 1)
		else:
			action = np.argmax(Q[:, s.dv, s.pv])

		N[action, s.dv, s.pv] = N[action, s.dv, s.pv] + 1
		path.append((action, s.dv, s.pv))
		s = step(s, action)

	for i in range(0, len(path)):
		# print(path[i])
		# path[i]
		# print(s.winner)
		alpha = 1 / (N[path[i][0], path[i][1], path[i][2]])
		# print(alpha)
		# print("V before", V[path[i][0], path[i][1], path[i][2]])
		Q[path[i][0], path[i][1], path[i][2]] = Q[path[i][0], path[i][1], path[i][2]] + alpha * (s.winner - Q[path[i][0], path[i][1], path[i][2]])
		# print("V after ", V[path[i][0], path[i][1], path[i][2]])


def playMC(n, plot=True):
	Q = np.zeros((2, 11, 22))
	N = np.zeros((2, 11, 22))
	for i in range(0, n):
		# print("game number", i)
		# create state
		# s = State()
		# recursiveMC(s, V, N, 100)
		MCgame(Q, N)

	print(Q[0, 1:11, 1:22])
	print("\n")
	# print(V[1, :, :])
	V = np.maximum(Q[0, 1:11, 1:22], Q[1, 1:11, 1:22])
	if(plot is True):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_ylim(11, 1)
		y = np.linspace(1, 11, 10)
		x = np.linspace(1, 21, 21)
		X, Y = np.meshgrid(x, y)
		surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,antialiased=True,shade=True)
		plt.show()
	return Q


def playMC_r(n):
	Q = np.zeros((2, 11, 22))
	N = np.zeros((2, 11, 22))
	for i in range(0, n):
		# print("game number", i)
		# create state
		s = State()
		recursiveMC(s, Q, N, 100)

	print(Q[0, 1:11, 1:22])
	print("\n")
	# print(V[1, :, :])
	V = np.maximum(Q[0, 1:11, 1:22], Q[1, 1:11, 1:22])
	# print(N)
	print((Q[1, 0, :]))
	print((Q[1, :, 0]))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_ylim(11, 1)
	y = np.linspace(1, 11, 10)
	x = np.linspace(1, 21, 21)
	X, Y = np.meshgrid(x, y)
	surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,antialiased=True,shade=True)
	plt.show()


	# print(V[0, :, 19])
	# print(V[0, :, 20])
	# print(V[0, :, 21])


def sarsa(n, lambd, plot=True):
	Q = np.zeros((2, 11, 22))
	N = np.zeros((2, 11, 22))
	for i in range(0, n):
		sarsaEpisode(Q, N, 1, lambd)
	# print(Q[0, 1:11, 1:22])
	# print("\n")
	# print(Q[1, 1:11, 1:22])
	# print("\n")
	# print(N[0, 1:11, 1:22])
	# print("\n")
	# print(N[1, 1:11, 1:22])
	V = np.maximum(Q[0, 1:11, 1:22], Q[1, 1:11, 1:22])
	if(plot is True):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_ylim(11, 1)
		y = np.linspace(1, 11, 10)
		x = np.linspace(1, 21, 21)
		X, Y = np.meshgrid(x, y)
		surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,antialiased=True,shade=True)
		plt.show()
	return Q


def epsilon_g_policy(s, N, Q):
	N0 = 100
	epsilon = N0 / (N0 + (N[0, s.dv, s.pv] + N[1, s.dv, s.pv]))
	if(np.random.uniform() < epsilon):  # choose randomly
		action = random.randint(0, 1)
	else:
		action = np.argmax(Q[:, s.dv, s.pv])

	return action


def sarsaEpisode(Q, N, gamma, lambd):
	# initialize Eligability matrix to 0 for eaxh episode
	E = np.zeros((2, 11, 22))
	path = []
	# Initialize first action and first state
	s = State()
	action = epsilon_g_policy(s, N, Q)
	while(s.isEnd() is False):
		new_state = step(copy.deepcopy(s), action)
		if(new_state.isEnd() is False):
			new_action = epsilon_g_policy(new_state, N, Q)
			delta = new_state.winner + gamma * Q[new_action, new_state.dv, new_state.pv] - Q[action, s.dv, s.pv]
		else:
			delta = new_state.winner - Q[action, s.dv, s.pv]
		E[action, s.dv, s.pv] = E[action, s.dv, s.pv] + 1
		N[action, s.dv, s.pv] = N[action, s.dv, s.pv] + 1
		path.append((action, s.dv, s.pv))
		for i in range(0, len(path)):
			alpha = 1 / (N[path[i][0], path[i][1], path[i][2]])
			Q[path[i][0], path[i][1], path[i][2]] = Q[path[i][0], path[i][1], path[i][2]] + alpha * delta * E[path[i][0], path[i][1], path[i][2]]
			E[path[i][0], path[i][1], path[i][2]] = gamma * lambd * E[path[i][0], path[i][1], path[i][2]]

		if(new_state.isEnd() is False):
			action = new_action
		s = new_state


def calcMSE(Q, Q_star):
	mse_sum = 0
	for j in range(0, 2):
		for k in range(0, 11):
			for l in range(0, 22):
				mse_sum = mse_sum + ((Q[j, k, l] - Q_star[j, k, l]) ** 2)
	return mse_sum / (2 * 10 * 21)


def msePlot(Q_star):
	mse_vals = []
	for i in range(0, 11):
		lambd = i * 0.1
		# for j in range(1, 1000):
		Q = sarsa(1000, lambd, False)
		if(i == 0 or i == 1):
			pass
		mse_vals.append(calcMSE(Q, Q_star))
	print(mse_vals)
	x = np.linspace(0, 1, 11)

	plt.plot(x, mse_vals)
	plt.xlabel('Lambda')
	plt.ylabel('MSE')
	plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
	plt.title("Mean-squared error plotted against lambda values")

	# plt.legend()

	plt.show()
	return Q


def msePlot_episode(Q_star):
	mse_vals = np.zeros((2, 1000))
	for j in range(0, 2):
		Q = np.zeros((2, 11, 22))
		N = np.zeros((2, 11, 22))
		for i in range(0, 1000):
			sarsaEpisode(Q, N, 1, j)
			mse_vals[j][i] = calcMSE(Q, Q_star)
	x = np.linspace(0, 1000, 1000)
	plt.plot(x, mse_vals[0], label="lambda = 0")
	plt.plot(x, mse_vals[1], label="lambda = 1")
	plt.xlabel('Episode')
	plt.ylabel('MSE')
	# plt.xticks(np.arange(min(x), max(x) + 1, 100))
	# plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
	plt.title("Mean-squared error plotted against episodes")

	plt.legend()

	plt.show()


# x = Card()
# x = playgames(20)
# print(x)
# print("player won", len(x[x == 1]))
# print("dealer won", len(x[x == -1]))
# print("draw was", len(x[x == 0]))

# s1 = State()
# print(Player().playRandom(s1, 30))
# print(np.zeros((21, 10)))

q_star = playMC(100000, False)
q = msePlot(q_star)
# q2 = msePlot_episode(q_star)
# playMC_r(100000)
# sarsa(10, 0.5)

# a = 0 + 1 * (1 - 0)
# b = a + 0.5 * (1 - a)
# sarsa(30000)
