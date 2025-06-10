import numpy as np
import gym
from src.agent import DQNAgent
from src.environment import create_env
import os
import cv2

EPISODES = 100
SAVE_INTERVAL = [0, 50]

def train():
    env = create_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e in SAVE_INTERVAL or e == EPISODES - 1:
            agent.model.save(f"artifacts/cartpole_dqn_episode_{e}.h5")
            agent.target_model.save(f"artifacts/cartpole_dqn_episode_{e}_target.h5")

    agent.model.save("artifacts/cartpole_dqn_final.h5")
    agent.target_model.save("artifacts/cartpole_dqn_final_target.h5")

if __name__ == '__main__':
    train()
