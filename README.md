[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/using-DeepQ-a83432.svg)](https://pypi.org/project/DeepQ/)
## DeepQ
A reinforcement learning library in Python.

This is a basic reinforcement learning library that works with **gym** and **tensorflow**. It uses a reinforcement learning
approach to machine learning. This is where the program is rewarded if it does the correct thing and if it does the wrong thing
it's punished

To install with [pip] do `pip install DeepQ`
. The other dependencies you need are **tensorflow** and **numpy**

### Example
<p align="center">
    <img src="https://i.ibb.co/HHd2WNZ/ezgif-com-gif-maker.gif" width=200 alt="video" border="0">
</p>

[example.py] this uses gym (which is a aim training lib), for this example i am using the environment 'LunarLander-v2' which simulates landing a 
spacecraft on the moon. We then give control to the AI that uses DeepQ's Agent(the spacecraft) which then learns to land it!:

```PY
from DeepQ import Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')  # loads the lunar lander trainer from gym
    learning_rate = 0.001
    n_games = 500  # this is the number of games to loops through
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
```

[pip]:https://pypi.org/project/DeepQ/
[example.py]:https://github.com/Madmegsox1/DeepQ/blob/main/example.py

## Documentation

The documentation [_**link**_]:



##### For more info Dm on Discord **Madmeg#4882**


[_**link**_]:https://github.com/Madmegsox1/DeepQ/blob/main/docs/agent.md
