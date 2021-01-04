import numpy as np
import matplotlib.pyplot as plt
import gym

env = gym.make('MountainCar-v0')

def selectAction(state, Q, epsilon):
	if (np.random.random() < epsilon):
		action = np.random.randint(0, env.action_space.n)
	else:
		action = np.argmax([Q[state[0], state[1]] for a in range(env.action_space.n)])
	return action

def selectBestAction(state, Q):
	return np.argmax([Q[state[0], state[1]] for a in range(env.action_space.n)])

def getMaxQ(state, Q):
	return np.max([Q[state[0], state[1]] for a in range(env.action_space.n)])

def discretizeState(state):
	return (np.round((state - env.observation_space.low) * [10, 100])).astype(int)

def QLearning(episodes, epsilon, alpha, discount):
	nb_states = (np.round((env.observation_space.high - env.observation_space.low) * [10, 100]) + 1).astype(int)
	Q = np.zeros((nb_states[0], nb_states[1], env.action_space.n))

	rewards = []
	meanRewards = []

	for i in range(episodes):
		done = False
		state = discretizeState(env.reset())
		inReward = 0

		while (not done):
			#env.render()
			action = selectAction(state, Q, epsilon)

			newState, reward, done, info = env.step(action)
			newState = discretizeState(newState)

			delta = alpha * (reward + discount * getMaxQ(newState, Q) - Q[state[0], state[1], action])
			Q[state[0], state[1], action] += delta

			state = newState
			inReward += reward

		rewards.append(inReward)
		if (i % 100 == 0):
			currentMean = np.mean(rewards)
			rewards = []
			meanRewards.append(currentMean)

	return Q, meanRewards

def quickDemo(episodes, Q):
	for i in range(episodes):
		done = False
		state = discretizeState(env.reset())

		while (not done):
			env.render()
			action = selectBestAction(state, Q)

			newState, reward, done, info = env.step(action)
			newState = discretizeState(newState)

			state = newState

Q, rewards = QLearning(5000, 0.05, 0.2, 0.9)

plt.plot(rewards)
plt.savefig('tmp_plots/mountainCar_rewards.png')

quickDemo(10, Q)
