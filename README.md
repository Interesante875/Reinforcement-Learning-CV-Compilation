# Reinforcement Learning (RL) Algorithms Notebook

## Overview

This Jupyter Notebook presents an in-depth implementation of various RL algorithms from scratch, including DQN (Deep Q-Network), DDQN (Double Deep Q-Network), DDPG (Deep Deterministic Policy Gradient), and Rainbow DQN.

## Approach

### 1. Markov Decision Process (MDP) Definition

Define the MDP as a tuple (S, A, P, R), where:
- S is the set of states.
- A is the set of actions.
- P is the state transition probability function.
- R is the reward function.

### 2. Q-Value Update Equation

For DQN and DDQN, the Q-value update is performed using the following equation:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right) $$

Where:
- \( Q(s, a) \) is the Q-value for state-action pair (s, a).
- \( \alpha \) is the learning rate.
- \( r \) is the immediate reward.
- \( \gamma \) is the discount factor.
- \( \max_{a'} Q(s', a') \) is the maximum Q-value for the next state.

### 3. Actor-Critic Architecture for DDPG

DDPG employs an actor-critic architecture with separate networks for the actor and critic. The actor network determines the policy, while the critic evaluates the Q-value. The actor is updated using the gradient of the Q-value with respect to the actions.

### 4. Rainbow DQN Components

Rainbow DQN combines several enhancements:
- Prioritized Experience Replay: Sampling experiences based on their priority.
- Double Q-Learning: Reducing overestimation of Q-values.
- Dueling Architecture: Separating the estimation of state value and advantage functions.
- Noisy Nets: Adding noise to the network parameters to encourage exploration.

## Equations

### 1. DDPG Policy Update

$$ \nabla_{\theta^\mu} J \approx \mathbb{E}_{s_t \sim \rho^\beta} \left[ \nabla_a Q(s, a | \theta^Q) |_{s=s_t, a=\mu(s_t | \theta^\mu)} \nabla_{\theta^\mu} \mu(s | \theta^\mu) \right] $$

Where:
- \( J \) is the expected return.
- \( \rho^\beta \) is a replay buffer.

### 2. Rainbow DQN Loss

$$ L(\theta_i) = \mathbb{E}_{(s, a, r, s', d) \sim U(D)} \left[ (y_i - Q(s, a | \theta_i))^2 \right] $$

Where:
- \( U(D) \) is the uniform random sample from the replay buffer.
- \( y_i \) is the target value.
# Zernike Subpixel Detection
## Overview

Zernike Subpixel Detection is a technique used in image processing for precise localization of object features. The notebook focuses on the implementation and analysis of the Zernike moments for subpixel detection.

## Approach

### 1. Zernike Moments

Zernike moments are mathematical descriptors that capture the shape information of an object in an image. The approach involves the calculation of Zernike polynomials, which are then used to extract the Zernike moments. These moments serve as a robust representation of the object's shape, enabling accurate subpixel detection.

### 2. Subpixel Detection

Subpixel detection is crucial for achieving higher precision in locating object features. Zernike moments provide a unique advantage in subpixel detection by offering a continuous and smooth representation of the object's boundary. The approach involves leveraging the Zernike moments to refine the position of object features at a subpixel level.

### 3. Equations for Zernike Polynomials

The Zernike polynomials, denoted as \(Z_{nm}\), are defined on the unit circle as follows:

$$ Z_{nm}(r, \theta) = R_n^m(r) \cdot \cos(m \cdot \theta) $$

Where:
- \( n \) and \( m \) are integers with \( n \geq m \geq 0 \).
- \( r \) is the radial coordinate ranging from 0 to 1.
- \( \theta \) is the azimuthal angle ranging from 0 to \( 2\pi \).
- \( R_n^m(r) \) is the radial polynomial given by the recurrence relation.

### 4. Subpixel Detection Formula

The subpixel detection involves refining the position of object features by considering the intensity distribution. The refined position \( (x', y') \) can be calculated using the weighted average of neighboring pixel positions:

$$ x' = \frac{\sum_{i,j} I(i, j) \cdot i}{\sum_{i,j} I(i, j)} $$

$$ y' = \frac{\sum_{i,j} I(i, j) \cdot j}{\sum_{i,j} I(i, j)} $$

Where:
- \( I(i, j) \) is the intensity value at pixel \((i, j)\).
