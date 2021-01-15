from DeepQ import Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')  # loads the lunar lander trainer from gym
    learning_rate = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=learning_rate,
                  input_dims=env.observation_space.shape, n_actions=env.action_space.n, mem_size=1000000,
                  batch_size=64, epsilon_end=0.01)
    scores = []
    epsilon_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        epsilon_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("Episode:", i, " Score %.2f" % score,
              "Average score %.2f" % avg_score,
              "epsilon %.2f" % agent.epsilon)

    plt.plot(scores)
    plt.title("A graph of the score increase over each episode of learning 'LunarLander-v2'")
    plt.ylabel("score")
    plt.xlabel("Episode")
    plt.show()  # plots the scores
