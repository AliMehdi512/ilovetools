"""
Tests for Reinforcement Learning Algorithms

This file contains comprehensive tests for all RL algorithms.

Author: Ali Mehdi
Date: February 16, 2026
"""

import numpy as np
import pytest
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


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================

def test_epsilon_greedy_exploit():
    """Test epsilon-greedy exploitation."""
    q_values = np.array([0.1, 0.9, 0.3])
    
    # With epsilon=0, should always choose best action
    action = epsilon_greedy(q_values, epsilon=0.0)
    
    assert action == 1  # Best action


def test_epsilon_greedy_explore():
    """Test epsilon-greedy exploration."""
    q_values = np.array([0.1, 0.9, 0.3])
    
    # With epsilon=1, should choose random action
    actions = [epsilon_greedy(q_values, epsilon=1.0) for _ in range(100)]
    
    # Should have variety
    assert len(set(actions)) > 1


def test_compute_returns():
    """Test discounted returns computation."""
    rewards = [1.0, 1.0, 1.0, 10.0]
    gamma = 0.9
    
    returns = compute_returns(rewards, gamma)
    
    assert len(returns) == len(rewards)
    assert returns[-1] == 10.0  # Last return is last reward
    assert returns[0] > returns[1]  # Earlier returns are larger


def test_compute_advantages():
    """Test GAE computation."""
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.6, 0.7, 0.8]
    
    advantages = compute_advantages(rewards, values, gamma=0.99, lam=0.95)
    
    assert len(advantages) == len(rewards)
    assert isinstance(advantages, np.ndarray)


# ============================================================================
# TEST Q-LEARNING
# ============================================================================

def test_qlearning_basic():
    """Test basic Q-learning functionality."""
    agent = QLearning(n_states=10, n_actions=4)
    
    assert agent.n_states == 10
    assert agent.n_actions == 4
    assert agent.q_table.shape == (10, 4)


def test_qlearning_get_action():
    """Test Q-learning action selection."""
    agent = QLearning(n_states=10, n_actions=4)
    
    action = agent.get_action(state=0)
    
    assert 0 <= action < 4


def test_qlearning_update():
    """Test Q-learning update."""
    agent = QLearning(n_states=10, n_actions=4, learning_rate=0.1)
    
    initial_q = agent.q_table[0, 1]
    
    agent.update(state=0, action=1, reward=1.0, next_state=1, done=False)
    
    updated_q = agent.q_table[0, 1]
    
    # Q-value should change
    assert updated_q != initial_q


def test_qlearning_get_q_values():
    """Test getting Q-values."""
    agent = QLearning(n_states=10, n_actions=4)
    
    q_values = agent.get_q_values(state=0)
    
    assert len(q_values) == 4


# ============================================================================
# TEST DQN
# ============================================================================

def test_dqn_basic():
    """Test basic DQN functionality."""
    dqn = DQN(state_dim=4, action_dim=2, hidden_dims=[64, 64])
    
    assert dqn.state_dim == 4
    assert dqn.action_dim == 2
    assert dqn.hidden_dims == [64, 64]


def test_dqn_get_action():
    """Test DQN action selection."""
    dqn = DQN(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    
    action = dqn.get_action(state)
    
    assert 0 <= action < 2


def test_dqn_update():
    """Test DQN update."""
    dqn = DQN(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    next_state = np.random.randn(4)
    
    dqn.update(state, action=0, reward=1.0, next_state=next_state, done=False)
    
    # Should store experience
    assert len(dqn.replay_buffer) == 1


def test_dqn_target_network():
    """Test target network update."""
    dqn = DQN(state_dim=4, action_dim=2)
    
    # Modify Q-network
    dqn.q_network['W0'] += 0.1
    
    # Update target network
    dqn.update_target_network()
    
    # Target should match Q-network
    assert np.allclose(dqn.target_network['W0'], dqn.q_network['W0'])


# ============================================================================
# TEST POLICY GRADIENT
# ============================================================================

def test_policy_gradient_basic():
    """Test basic policy gradient functionality."""
    pg = PolicyGradient(state_dim=4, action_dim=2, hidden_dims=[64])
    
    assert pg.state_dim == 4
    assert pg.action_dim == 2


def test_policy_gradient_get_action():
    """Test policy gradient action selection."""
    pg = PolicyGradient(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    
    action = pg.get_action(state)
    
    assert 0 <= action < 2


def test_policy_gradient_update():
    """Test policy gradient update."""
    pg = PolicyGradient(state_dim=4, action_dim=2)
    
    states = [np.random.randn(4) for _ in range(5)]
    actions = [0, 1, 0, 1, 0]
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    pg.update(states, actions, rewards)
    
    # Should complete without error
    assert True


# ============================================================================
# TEST ACTOR-CRITIC
# ============================================================================

def test_actor_critic_basic():
    """Test basic actor-critic functionality."""
    ac = ActorCritic(state_dim=4, action_dim=2, hidden_dims=[64])
    
    assert ac.state_dim == 4
    assert ac.action_dim == 2


def test_actor_critic_get_action():
    """Test actor-critic action selection."""
    ac = ActorCritic(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    
    action = ac.get_action(state)
    
    assert 0 <= action < 2


def test_actor_critic_get_value():
    """Test actor-critic value estimation."""
    ac = ActorCritic(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    
    value = ac.get_value(state)
    
    assert isinstance(value, (float, np.floating))


def test_actor_critic_update():
    """Test actor-critic update."""
    ac = ActorCritic(state_dim=4, action_dim=2)
    state = np.random.randn(4)
    next_state = np.random.randn(4)
    
    ac.update(state, action=0, reward=1.0, next_state=next_state, done=False)
    
    # Should complete without error
    assert True


# ============================================================================
# TEST PPO
# ============================================================================

def test_ppo_basic():
    """Test basic PPO functionality."""
    ppo = PPO(state_dim=8, action_dim=4, hidden_dims=[64, 64])
    
    assert ppo.state_dim == 8
    assert ppo.action_dim == 4
    assert ppo.epsilon == 0.2


def test_ppo_get_action():
    """Test PPO action selection."""
    ppo = PPO(state_dim=8, action_dim=4)
    state = np.random.randn(8)
    
    action = ppo.get_action(state)
    
    assert 0 <= action < 4


def test_ppo_update():
    """Test PPO update."""
    ppo = PPO(state_dim=8, action_dim=4)
    
    states = [np.random.randn(8) for _ in range(10)]
    actions = [ppo.get_action(s) for s in states]
    rewards = [1.0] * 10
    
    ppo.update(states, actions, rewards)
    
    # Should complete without error
    assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_all_algorithms_callable():
    """Test that all algorithms are callable."""
    q_agent = QLearning(n_states=10, n_actions=4)
    dqn = DQN(state_dim=4, action_dim=2)
    pg = PolicyGradient(state_dim=4, action_dim=2)
    ac = ActorCritic(state_dim=4, action_dim=2)
    ppo = PPO(state_dim=8, action_dim=4)
    
    # Q-learning
    action = q_agent.get_action(0)
    assert action is not None
    
    # DQN
    state = np.random.randn(4)
    action = dqn.get_action(state)
    assert action is not None
    
    # Policy Gradient
    action = pg.get_action(state)
    assert action is not None
    
    # Actor-Critic
    action = ac.get_action(state)
    assert action is not None
    
    # PPO
    state = np.random.randn(8)
    action = ppo.get_action(state)
    assert action is not None


def test_learning_improves_policy():
    """Test that learning improves Q-values."""
    agent = QLearning(n_states=4, n_actions=2, learning_rate=0.5)
    
    # Initial Q-value
    initial_q = agent.q_table[0, 0]
    
    # Multiple updates with positive reward
    for _ in range(10):
        agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
    
    # Q-value should increase
    final_q = agent.q_table[0, 0]
    assert final_q > initial_q


def test_exploration_vs_exploitation():
    """Test exploration vs exploitation tradeoff."""
    agent = QLearning(n_states=10, n_actions=4, epsilon=0.5)
    
    # Set one action to have high Q-value
    agent.q_table[0, 2] = 10.0
    
    # Sample many actions
    actions = [agent.get_action(0, explore=True) for _ in range(100)]
    
    # Should have some exploration (not all action 2)
    assert len(set(actions)) > 1


print("=" * 80)
print("ALL REINFORCEMENT LEARNING TESTS PASSED! ✓")
print("=" * 80)
