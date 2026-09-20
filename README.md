# 🕹️ Deep Q-Network (DQN) & SAC for CartPole Control

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-brightgreen.svg)
![Reinforcement Learning](https://img.shields.io/badge/Domain-Reinforcement%20Learning-purple.svg)

This repository contains a from-scratch implementation of a Deep Q-Network (DQN) agent and a Soft Actor-Critic (SAC) algorithm designed to solve the classic CartPole-v1 control problem. The project demonstrates a fundamental understanding of model-free reinforcement learning, transitioning from value-based methods to state-of-the-art actor-critic approaches.

## 📌 Project Overview

Developing stable reinforcement learning agents requires careful orchestration of neural networks, environment interactions, and memory management. This project tackles the CartPole-v1 balancing task by implementing two distinct algorithms:
1.  **DQN (Value-Based):** Learns an optimal action-value function utilizing an epsilon-greedy exploration strategy.
2.  **SAC (Actor-Critic):** A maximum entropy approach utilizing a stochastic policy to balance exploration and exploitation automatically.

## ✨ Key Features & Implementations

*   **Custom Deep Q-Network (DQN):** 
    *   Engineered a Dense Neural Network (`FCModel` / `QNetwork`) to approximate Q-values for high-dimensional state inputs.
    *   Integrated a **Target Network** with periodic updates to stabilize the temporal difference (TD) learning process.
    *   Implemented an **Epsilon-Greedy Policy** with decay to systematically manage the exploration-exploitation trade-off.
*   **Experience Replay Buffer:** Built a `ReplayMemory` module from scratch with burn-in functionality. This breaks correlations in consecutive training data and enables efficient mini-batch learning.
*   **Soft Actor-Critic (SAC) Extension:** 
    *   Implemented an Actor network for stochastic policy generation using the reparameterization trick.
    *   Developed Twin Critic networks to minimize overestimation bias.
    *   Applied soft target updates (Polyak averaging, $\tau$) and entropy regularization ($\alpha$).
*   **Performance Evaluation:** Conducted robust testing across 5 random seeds, visualizing the agent's learning progress via mean returns and standard deviation confidence intervals.

## 🧠 Theoretical Insights

The accompanying notebook explores core Reinforcement Learning concepts:
*   **Model-Free Learning:** The agents learn strictly from $(S, A, R, S')$ tuples without attempting to model environmental transition dynamics.
*   **Value vs. Policy Methods:** Highlights the differences between implicitly deriving a policy from a learned Q-function (DQN) versus directly optimizing a parameterized stochastic policy (SAC).
*   **Replay Buffer Dynamics:** Demonstrates how random sampling from past experiences prevents short-term memory bias and increases sample efficiency.

## 🛠️ Technical Stack

*   **Core Logic:** Python
*   **Deep Learning Framework:** PyTorch (`torch.nn`, `torch.optim`)
*   **Simulation Environment:** Gymnasium (`CartPole-v1`)
*   **Data & Visualization:** NumPy, Matplotlib, Tqdm

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the following dependencies:

```bash
pip install torch gymnasium numpy matplotlib tqdm
