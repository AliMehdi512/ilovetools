"""
Comprehensive Examples: Reinforcement Learning Algorithms

This file demonstrates all RL algorithms with practical examples.

Author: Ali Mehdi
Date: February 16, 2026
"""

import numpy as np
from ilovetools.ml.reinforcement import (
    QLearning,
    DQN,
    PolicyGradient,
    ActorCritic,
    PPO,
    epsilon_greedy,
    compute_returns,
    compute_advantages,
)

print("=" * 80)
print("REINFORCEMENT LEARNING ALGORITHMS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Q-Learning - Grid World
# ============================================================================
print("EXAMPLE 1: Q-Learning - Grid World Navigation")
print("-" * 80)

# 4x4 grid world
n_states = 16  # 4x4 grid
n_actions = 4  # up, down, left, right

agent = QLearning(
    n_states=n_states,
    n_actions=n_actions,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1
)

print("Q-Learning configuration:")
print(f"States: {n_states} (4x4 grid)")
print(f"Actions: {n_actions} (up, down, left, right)")
print(f"Learning rate: {agent.lr}")
print(f"Discount factor: {agent.gamma}")
print(f"Exploration rate: {agent.epsilon}")
print()

# Simulate training
print("Training for 100 episodes...")
for episode in range(100):
    state = 0  # Start state
    
    for step in range(20):
        # Select action
        action = agent.get_action(state)
        
        # Simulate environment
        next_state = (state + 1) % n_states
        reward = 1.0 if next_state == 15 else 0.0  # Goal at state 15
        done = (next_state == 15)
        
        # Update Q-table
        agent.update(state, action, reward, next_state, done)
        
        if done:
            break
        
        state = next_state

print(f"Training completed!")
print(f"Q-table shape: {agent.q_table.shape}")
print(f"Best action at start: {agent.get_action(0, explore=False)}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: DQN - Continuous State Space
# ============================================================================
print("EXAMPLE 2: DQN - CartPole-like Environment")
print("-" * 80)

state_dim = 4  # Cart position, velocity, pole angle, angular velocity
action_dim = 2  # Left, right

dqn = DQN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[64, 64],
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1
)

print("DQN configuration:")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Hidden layers: {dqn.hidden_dims}")
print(f"Replay buffer size: {dqn.replay_buffer.maxlen}")
print()

# Simulate training
print("Training for 50 episodes...")
for episode in range(50):
    state = np.random.randn(state_dim)
    
    for step in range(100):
        # Select action
        action = dqn.get_action(state)
        
        # Simulate environment
        next_state = np.random.randn(state_dim)
        reward = 1.0
        done = (step == 99)
        
        # Store experience
        dqn.update(state, action, reward, next_state, done)
        
        if done:
            break
        
        state = next_state
    
    # Update target network every 10 episodes
    if episode % 10 == 0:
        dqn.update_target_network()

print(f"Training completed!")
print(f"Replay buffer size: {len(dqn.replay_buffer)}")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Policy Gradient - REINFORCE
# ============================================================================
print("EXAMPLE 3: Policy Gradient - REINFORCE Algorithm")
print("-" * 80)

state_dim = 4
action_dim = 2

pg = PolicyGradient(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[64],
    learning_rate=0.001,
    gamma=0.99
)

print("Policy Gradient configuration:")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Hidden layers: {pg.hidden_dims}")
print()

# Simulate one episode
print("Running one episode...")
states = []
actions = []
rewards = []

state = np.random.randn(state_dim)

for step in range(20):
    # Sample action from policy
    action = pg.get_action(state)
    
    # Simulate environment
    next_state = np.random.randn(state_dim)
    reward = 1.0
    
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    
    state = next_state

print(f"Episode length: {len(states)}")
print(f"Total reward: {sum(rewards)}")
print()

# Update policy
pg.update(states, actions, rewards)

print("Policy updated using REINFORCE!")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Actor-Critic - Advantage Actor-Critic
# ============================================================================
print("EXAMPLE 4: Actor-Critic - Advantage Actor-Critic")
print("-" * 80)

state_dim = 4
action_dim = 2

ac = ActorCritic(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[64],
    learning_rate=0.001,
    gamma=0.99
)

print("Actor-Critic configuration:")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Actor network: {ac.actor.keys()}")
print(f"Critic network: {ac.critic.keys()}")
print()

# Simulate training
print("Training for 30 episodes...")
for episode in range(30):
    state = np.random.randn(state_dim)
    
    for step in range(50):
        # Get action from actor
        action = ac.get_action(state)
        
        # Get value from critic
        value = ac.get_value(state)
        
        # Simulate environment
        next_state = np.random.randn(state_dim)
        reward = 1.0
        done = (step == 49)
        
        # Update actor and critic
        ac.update(state, action, reward, next_state, done)
        
        if done:
            break
        
        state = next_state

print(f"Training completed!")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: PPO - State-of-the-Art
# ============================================================================
print("EXAMPLE 5: PPO - Proximal Policy Optimization")
print("-" * 80)

state_dim = 8
action_dim = 4

ppo = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[64, 64],
    learning_rate=0.0003,
    gamma=0.99,
    epsilon=0.2,
    lam=0.95
)

print("PPO configuration:")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Hidden layers: {ppo.hidden_dims}")
print(f"Clipping epsilon: {ppo.epsilon}")
print(f"GAE lambda: {ppo.lam}")
print()

# Simulate training
print("Training for 20 episodes...")
for episode in range(20):
    states = []
    actions = []
    rewards = []
    
    state = np.random.randn(state_dim)
    
    for step in range(100):
        # Sample action
        action = ppo.get_action(state)
        
        # Simulate environment
        next_state = np.random.randn(state_dim)
        reward = 1.0
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Update policy
    ppo.update(states, actions, rewards)

print(f"Training completed!")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Epsilon-Greedy Exploration
# ============================================================================
print("EXAMPLE 6: Epsilon-Greedy Exploration Strategy")
print("-" * 80)

q_values = np.array([0.1, 0.9, 0.3, 0.5])

print("Q-values: [0.1, 0.9, 0.3, 0.5]")
print()

# Pure exploitation
print("Pure exploitation (epsilon=0):")
action = epsilon_greedy(q_values, epsilon=0.0)
print(f"  Selected action: {action} (best action)")
print()

# Balanced exploration-exploitation
print("Balanced (epsilon=0.1):")
actions = [epsilon_greedy(q_values, epsilon=0.1) for _ in range(100)]
print(f"  Action distribution: {np.bincount(actions, minlength=4)}")
print()

# Pure exploration
print("Pure exploration (epsilon=1.0):")
actions = [epsilon_greedy(q_values, epsilon=1.0) for _ in range(100)]
print(f"  Action distribution: {np.bincount(actions, minlength=4)}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Discounted Returns
# ============================================================================
print("EXAMPLE 7: Discounted Returns Computation")
print("-" * 80)

rewards = [1.0, 1.0, 1.0, 10.0]

print(f"Rewards: {rewards}")
print()

# Different discount factors
for gamma in [0.9, 0.95, 0.99]:
    returns = compute_returns(rewards, gamma)
    print(f"Gamma={gamma}: Returns={returns}")

print()
print("Interpretation:")
print("✓ Higher gamma → More weight on future rewards")
print("✓ Lower gamma → More weight on immediate rewards")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Generalized Advantage Estimation (GAE)
# ============================================================================
print("EXAMPLE 8: Generalized Advantage Estimation (GAE)")
print("-" * 80)

rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"Rewards: {rewards}")
print(f"Values: {values}")
print()

# Different lambda values
for lam in [0.0, 0.5, 0.95, 1.0]:
    advantages = compute_advantages(rewards, values, gamma=0.99, lam=lam)
    print(f"Lambda={lam}: Advantages={advantages}")

print()
print("Interpretation:")
print("✓ Lambda=0: One-step TD (low variance, high bias)")
print("✓ Lambda=1: Monte Carlo (high variance, low bias)")
print("✓ Lambda=0.95: Balanced (PPO default)")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing RL Algorithms
# ============================================================================
print("EXAMPLE 9: Comparing RL Algorithms")
print("-" * 80)

print("Algorithm comparison:")
print()

print("Q-Learning:")
print("  Type: Value-based (tabular)")
print("  Speed: ★★★★★ (very fast)")
print("  Sample efficiency: ★★★☆☆")
print("  Use case: Small discrete state/action spaces")
print()

print("DQN:")
print("  Type: Value-based (deep)")
print("  Speed: ★★★☆☆")
print("  Sample efficiency: ★★★★☆")
print("  Use case: High-dimensional states, discrete actions")
print()

print("Policy Gradient (REINFORCE):")
print("  Type: Policy-based")
print("  Speed: ★★☆☆☆")
print("  Sample efficiency: ★★☆☆☆")
print("  Use case: Continuous actions, stochastic policies")
print()

print("Actor-Critic:")
print("  Type: Hybrid")
print("  Speed: ★★★★☆")
print("  Sample efficiency: ★★★★☆")
print("  Use case: Continuous control, faster than PG")
print()

print("PPO:")
print("  Type: Policy-based (state-of-the-art)")
print("  Speed: ★★★★★")
print("  Sample efficiency: ★★★★★")
print("  Use case: Production RL, robotics, complex tasks")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Real-World Application - Robot Control
# ============================================================================
print("EXAMPLE 10: Real-World Application - Robot Arm Control")
print("-" * 80)

print("Robot arm control with PPO:")
print()

# State: joint angles, velocities (8D)
# Actions: joint torques (4D)
state_dim = 8
action_dim = 4

ppo = PPO(state_dim=state_dim, action_dim=action_dim)

print("Step 1: Initialize environment")
print(f"  State: {state_dim}D (joint angles + velocities)")
print(f"  Actions: {action_dim}D (joint torques)")
print()

print("Step 2: Training loop")
print("  For each episode:")
print("    1. Reset robot to initial position")
print("    2. Collect trajectory (states, actions, rewards)")
print("    3. Compute advantages using GAE")
print("    4. Update policy with PPO clipping")
print()

print("Step 3: Deployment")
print("  ✓ Load trained policy")
print("  ✓ Run in real-time on robot")
print("  ✓ No exploration (greedy policy)")
print()

print("Benefits of PPO:")
print("✓ Stable training (clipping prevents large updates)")
print("✓ Sample efficient (reuses data)")
print("✓ Works with continuous actions")
print("✓ State-of-the-art performance")

print("\n✓ Example 10 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Q-Learning - Grid World")
print("2. ✓ DQN - Continuous States")
print("3. ✓ Policy Gradient - REINFORCE")
print("4. ✓ Actor-Critic - Advantage AC")
print("5. ✓ PPO - State-of-the-Art")
print("6. ✓ Epsilon-Greedy Exploration")
print("7. ✓ Discounted Returns")
print("8. ✓ GAE Computation")
print("9. ✓ Algorithm Comparison")
print("10. ✓ Robot Control Application")
print()
print("You now have a complete understanding of reinforcement learning!")
print()
print("Next steps:")
print("- Use Q-Learning for simple discrete tasks")
print("- Use DQN for high-dimensional discrete tasks")
print("- Use PPO for continuous control and production RL")
print("- Apply to robotics, games, autonomous systems")
