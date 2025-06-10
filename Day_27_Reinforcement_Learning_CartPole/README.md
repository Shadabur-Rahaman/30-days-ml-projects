# 🎮 Day 27: Reinforcement Learning with CartPole

## 🧠 Project Title
**Deep Q-Learning Agent for CartPole-v1 (OpenAI Gym)**

---

## 🔍 Overview

This project demonstrates how to train a reinforcement learning agent using **Deep Q-Learning (DQN)** to solve the classic **CartPole-v1** environment from OpenAI Gym.

In the CartPole task, the agent learns to balance a pole on a moving cart using discrete actions (move left or right) based on the state of the environment.

---

## 🗂️ Directory Structure

Day27_Reinforcement_Learning_CartPole/
├── data/
│ └── rewards_log.csv # Episode rewards log (optional)
│
├── images/
│ ├── training_rewards.png # Plot of episode rewards over time
│ └── cartpole_agent.gif # Optional animated visualization
│
├── videos/
│ ├── cartpole_agent.mp4 # Trained agent's gameplay video
│ ├── rl-video-episode-0.mp4 # Episode 0 video (sample)
│ └── rl-video-episode-0.meta.json # Metadata
│
├── artifacts/
│ ├── cartpole_dqn_episode_0.h5 # Weights after episode 0
│ ├── cartpole_dqn_episode_0_target.h5 # Target network after episode 0
│ ├── cartpole_dqn_episode_50.h5 # Mid-training weights
│ ├── cartpole_dqn_episode_50_target.h5 # Mid-training target
│ ├── cartpole_dqn_final.h5 # Final trained model
│ └── cartpole_dqn_final_target.h5 # Final target model
│
├── notebooks/
│ └── Day_27_Reinforcement_Learning_CartPole.ipynb # Main training and visualization notebook
│
├── src/
│ ├── environment.py # OpenAI Gym environment setup
│ ├── agent.py # DQN Agent logic
│ ├── train.py # Training loop
│ └── utils.py # Plotting, saving models, etc.
│
├── requirements.txt # Required Python packages
├── .gitignore # Ignore saved weights, logs, etc.
└── README.md # This file

yaml
Copy
Edit

---

## 🎯 Learning Outcomes

✅ Understand the basics of reinforcement learning  
✅ Implement a Deep Q-Network (DQN) using PyTorch  
✅ Apply epsilon-greedy exploration strategy  
✅ Use experience replay buffers for training stability  
✅ Visualize training curves and agent performance  
✅ Save trained models and record gameplay

---

## 📈 Visualizations

- `training_rewards.png` — Plot showing episode rewards over time  
- `cartpole_agent.mp4` — Recorded video of trained agent's performance

---

## 🧪 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Run the Notebook
bash
Copy
Edit
jupyter notebook notebooks/Day_27_Reinforcement_Learning_CartPole.ipynb
3. (Optional) Run the Python Script
bash
Copy
Edit
python src/train.py
🧠 Model Architecture: Deep Q-Network (DQN)
scss
Copy
Edit
Input (State Vector)
      ⬇️
Fully Connected Layer (128 units, ReLU)
      ⬇️
Fully Connected Layer (64 units, ReLU)
      ⬇️
Output Layer (2 Q-values: Left, Right)
🧠 Agent Components
Experience Replay: Stores transitions and samples mini-batches

Target Network: Stabilizes learning

Epsilon-Greedy Strategy: Balances exploration vs. exploitation

Huber Loss: Handles outlier Q-values

Model Saving: Periodic weight checkpoints

📌 Stay Tuned
🚀 This is Day 27 of my #30DaysMLProjects journey.
