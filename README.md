# Implementation of Value Iteration for Optimal Policy Computation using Gymnasium

---

# Aim

To implement the **Value Iteration** algorithm for solving a finite **Markov Decision Process (MDP)** using the Gymnasium **FrozenLake-v1** environment and compute the **optimal state-value function** and **optimal policy** based on the Bellman Optimality Equation.

---

# Problem Statement

Develop a Python program that applies the **Value Iteration** algorithm to the **FrozenLake-v1** environment provided by Gymnasium. The algorithm should iteratively update the value of each state until convergence and then derive the optimal policy that maximizes the expected cumulative reward.

---

# Software Requirements

- Python 3.x
- Gymnasium
- NumPy
- Jupyter Notebook / Google Colab / VS Code

---

# Environment Description

The **FrozenLake-v1** environment is a grid-world problem in which an agent must move from the **Start (S)** state to the **Goal (G)** while avoiding **Holes (H)**.

Grid Used:

```text
S F F F
F H F H
F F F H
H F F G
```

Where:

- **S** – Start State
- **F** – Frozen Surface (Safe)
- **H** – Hole (Terminal State)
- **G** – Goal State (Reward = 1)

The environment is **stochastic (is_slippery=True)**, meaning the intended action may not always be executed.

---

# MDP Representation

An MDP is represented as:

**MDP = (S, A, P, R, γ)**

Where:

- **S** = Set of states (16 states)
- **A** = {Left, Down, Right, Up}
- **P(s'|s,a)** = Transition probability
- **R(s,a,s')** = Reward function
- **γ = 0.99** = Discount factor

---

# Theory

**Value Iteration** is a Dynamic Programming algorithm used to compute the optimal value function of an MDP.

It repeatedly updates the value of each state using the **Bellman Optimality Equation**:

\[
V(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right]
\]

The iterations continue until the maximum change in the value function is smaller than a predefined threshold.

After convergence, the optimal policy is obtained by selecting the action that gives the highest expected value.

---

# Algorithm

1. Create the FrozenLake environment.
2. Initialize the value function of all states to zero.
3. Repeat until convergence:
   - Compute the value for every possible action.
   - Update each state's value using the Bellman Optimality Equation.
   - Calculate the maximum difference between old and new values.
4. Stop when the difference becomes less than the threshold.
5. Extract the optimal policy by selecting the action with the highest value for every state.
6. Display the optimal value function and policy.

---

# Python Program

```python
import gymnasium as gym
import numpy as np

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
```

```python

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    iteration = 0

    while True:

        delta = 0

        for s in range(n_states):

            action_values = np.zeros(n_actions)

            for a in range(n_actions):

                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:

                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            best_value = np.max(action_values)

            delta = max(delta, abs(best_value - V[s]))

            V[s] = best_value

        iteration += 1

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):

        action_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:

                action_values[a] += prob * (
                    reward + gamma * V[next_state]
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration
```

```python

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iterations = value_iteration(env)

```

```python

# -------------------------------------------------
# Display Output
# -------------------------------------------------

print("Name: VENKATANATHAN P R")
print("Register Number: 212223240173")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")

print(np.round(V.reshape(4,4),4))

action_symbols = {
    0:"L",
    1:"D",
    2:"R",
    3:"U"
}

policy_grid = np.array(
    [action_symbols[a] for a in policy]
).reshape(4,4)

print("\nOptimal Policy:")

print(policy_grid)

env.close()
```

---

# Output

<img width="509" height="355" alt="image" src="https://github.com/user-attachments/assets/ac544473-689c-48ae-b5ba-190c24ae71c3" />

---

# Result

```text
The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after convergence using the Bellman Optimality Equation.
```

---

# Inference

```text
From this experiment, it is observed that the Value Iteration algorithm efficiently computes the optimal value of every state by repeatedly applying the Bellman Optimality Equation. Once the value function converges, the optimal policy is extracted by selecting the action with the highest expected return. This demonstrates how Dynamic Programming can solve finite Markov Decision Processes and determine the best sequence of actions for an agent.
```

---

