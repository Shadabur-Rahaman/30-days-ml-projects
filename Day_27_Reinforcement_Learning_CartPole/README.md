Day 27: Reinforcement Learning with CartPole

🧠 Project Title

Deep Q-Learning Agent for CartPole-v1 (OpenAI Gym)

🔍 Overview

This project demonstrates how to train a reinforcement learning agent using Deep Q-Learning (DQN) to solve the CartPole-v1 environment from OpenAI Gym.

CartPole is a classic control problem where the goal is to balance a pole on a moving cart using discrete left/right actions.

📆 Directory Structure
Day27_Reinforcement_Learning_CartPole/
├── data/
│   └── rewards_log.csv                         # Episode rewards log (optional)
│
├── images/
│   ├── training_rewards.png                    # Plot of episode rewards over time
│   └── cartpole_agent.gif                      # Optional animated visualization
│
├── videos/
│   ├── cartpole_agent.mp4                      # Trained agent's gameplay video
│   ├── rl-video-episode-0.mp4                  # Episode 0 video (sample)
│   └── rl-video-episode-0.meta.json            # Metadata
│
├── artifacts/
│   ├── cartpole_dqn_episode_0.h5               # Weights after episode 0
│   ├── cartpole_dqn_episode_0_target.h5        # Target network after episode 0
│   ├── cartpole_dqn_episode_50.h5              # Mid-training weights
│   ├── cartpole_dqn_episode_50_target.h5       # Mid-training target
│   ├── cartpole_dqn_final.h5                   # Final trained model
│   └── cartpole_dqn_final_target.h5            # Final target model
│
├── notebooks/
│   └── Day_27_Reinforcement_Learning_CartPole.ipynb   # Main training and visualization notebook
│
├── src/
│   ├── environment.py                          # OpenAI Gym environment setup
│   ├── agent.py                                # DQN Agent logic
│   ├── train.py                                # Training loop
│   └── utils.py                                # Plotting, saving models, etc.
│
├── requirements.txt                            # Required Python packages
├── .gitignore                                  # Ignore saved weights, logs, etc.
└── README.md                                   # This file

🎯 Learning Outcomes

Understand the basics of reinforcement learning

Implement a Deep Q-Network (DQN) using PyTorch

Use epsilon-greedy exploration

Store and sample from experience replay buffers

Plot training curves and save model weights

Record and visualize agent gameplay

📈 Visualizations

training_rewards.png: Reward per episode over time

cartpole_agent.mp4: Final policy gameplay
```bash

⚙️ How to Run

1. Install Dependencies
pip install -r requirements.txt

2. Run the Notebook
jupyter notebook notebooks/Day_27_Reinforcement_Learning_CartPole.ipynb

3. (Optional) Run the Python script version
python src/train.py

🎮 Model Architecture (DQN)

Input (State)
   ⬇️
Fully Connected (128)
   ⬇️
Fully Connected (64)
   ⬇️
Output (Q-values for left/right)
