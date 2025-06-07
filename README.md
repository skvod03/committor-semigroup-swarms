

# Committor Update and Transition Rate Estimation

This repository implements an algorithm to estimate transition rates between two basins using a committor function, which quantifies the probability that a system will transition from one state (basin A) to another (basin B). The algorithm combines local sampling from each basin with importance sampling to provide accurate estimates of the committor function and the transition rates between basins. The algorithm has been applied to the Müller-Brown potential, a benchmark system in molecular dynamics.

## Computational Approach

The algorithm optimizes the committor function and estimates the reaction rate "on-the-fly". It combines multiple techniques, including simulation, deep learning, and importance sampling, to estimate the rates. The algorithm requires independent simulations of "swarm" trajectories, which are easily parallelizable.

### Sampling Process:

1. **Equilibrium Simulation**:

   * Initial equilibrium simulations are run in each basin (A and B) until the trajectories exit the basin a predefined number of times, $N_{\text{exit}}$.
   * These exit events are stored, including the trajectory configuration and exit time, forming an ensemble of points sampled from the basin boundaries $\chi_A$ and $\chi_B$.

2. **Importance Sampling**:

   * A configuration $\mathbf{x}_i$ is selected from the boundary ensemble of one basin.
   * A swarm of $k$ independent trajectories is initiated from this point and evolved. The trajectory endpoints are saved and used to train the committor function $q_\theta$.

3. **Committor Function Update**:

   * The committor $q_\theta$ is updated using Adam optimizer. The loss function used is a symmetrized discrepancy on the log-committor: 

$$ L_{n,k,\tau}(\theta) = \frac{1}{2n} \sum_{i=1}^{n} \left( \log\left(1 - q_\theta(x_i)\right) - \log\left( \frac{1}{k} \sum_{j=1}^{k} q_\theta(x_{i,j}^\tau) \right) \right)^2 $$

  This ensures that the committor is updated iteratively, considering both the forward and reverse transitions.

4. **Selection of Next Sample**:
   
   * The next sample $x_{i+1}$ is chosen as the endpoint of the swarm trajectory with the highest committor value under $q_\theta$.
   * This new sample is used to initiate the next swarm, and the committor is updated again. The process continues iteratively.


### Swarm-based Sampling:

The algorithm constructs a "chain" of samples connecting the basins. Swarms of independent trajectories are sampled from each basin, and the committor function is updated with each new trajectory. The next sampling point is selected from the set of previously sampled endpoints, with the highest committor value guiding the selection. This allows the exploration of alternative reaction pathways between the basins.

Here's an updated explanation for the loss function, its connection to the semigroup equation, and the application to the Müller-Brown potential:


### Loss Function Derivation and Connection to the Semigroup Equation

The loss function used in the algorithm is derived from the **semigroup equation** and is optimized to obtain the committor function $q_\theta(x)$ that describes the probability of transitioning from a starting point $x$ in one basin (say, Basin A) to the other basin (Basin B).

In the context of Langevin dynamics and transition rate estimation, the semigroup operator $P_\tau$ defines the evolution of the system over a time $\tau$, and can be seen as an expectation of a function $f$ evaluated at future states:

$$
P_\tau f(x_0) = \mathbb{E}[f(x_{\tau}) \mid x_0]
$$


Where $P_\tau$ is the **Markov semigroup** operator associated with the dynamics and $\mathbf{x}_0$ is an initial state in the configuration space $\Omega$. The function $f : \Omega \to \mathbb{R}$ is an observable in the state space.

The committor function $q(x_0)$ is defined as the probability that a system starting at state $x_0$ will transition to Basin B (from Basin A) before returning to Basin A. This probability can be written as the expectation of the committor function evaluated at the propagated state $x_0$:

$$
q(x_0) = \mathbb{E}[q(x_\tau)] \equiv P_\tau q(x_0)
$$

This is a Markovian relationship, and it captures the probabilistic nature of transitions between basins in the state space, considering the dynamics of the system under the Langevin equation.

### Loss Function Used in the Algorithm

To estimate the committor function, we use a **logarithmic discrepancy loss function** based on the semigroup equation. This is designed to optimize the committor while enforcing the boundary conditions of $q_\theta(x)$:

* $q_\theta(x \in A) = 0$ (starting in Basin A)
* $q_\theta(x \in B) = 1$ (starting in Basin B)

The loss function is given by:

$$
L(\theta) = \frac{1}{2} \int_{\Omega} \left( \log(q_\theta(x)) - \log(P_\tau q_\theta(x)) \right)^2 d\nu(x)
$$

Where:

* $\Omega$ is the configuration space.
* $\nu(x)$ is the sampling distribution (e.g., equilibrium distribution or the boundary ensemble).
* $P_\tau q_\theta(x)$ represents the propagated committor over a time $\tau$, as described in the semigroup equation.

This loss function reflects the discrepancy between the committor at the current state and its expected value at the next state, as determined by the propagator. This expectation ensures that the committor evolves as the system progresses through the state space.

### Algorithm and Sampling Procedure

The algorithm uses **importance sampling** and **swarm-based simulations** to sample trajectory points along reactive pathways between the two basins. These trajectories are used to update the committor function iteratively, ensuring that the committor captures the transition dynamics effectively. The key steps are:

1. **Equilibrium Simulations**: Run simulations in each basin (A and B) to collect data for the boundaries.
2. **Importance Sampling**: From the boundary configuration of one basin, generate multiple independent trajectories and evolve them toward the other basin. This allows for efficient sampling of the transition region.
3. **Committor Update**: Update the committor using the loss function based on both the initial boundary samples and the propagated points.

The process is repeated iteratively, with new samples being added to the training set and the committor updated at each step. The final result is a set of trajectory samples and an optimized committor function that accurately estimates transition probabilities.

### Application to the Müller-Brown Potential

The Müller-Brown potential is a classic two-basin system used to model molecular transitions, and it serves as the test case for this algorithm. The potential is given by:

$$
V(\mathbf{x}) = \sum_{i=1}^{4} A_i e^{-(\mathbf{x} - \mu_i)^T \Sigma_i^{-1} (\mathbf{x} - \mu_i)}
$$

Where:

* $A_i$ are the parameters defining the depth of the wells.
* $\mu_i$ are the centers of the wells.
* $\Sigma_i$ are the covariance matrices that define the shape of the wells.

This potential defines two basins (A and B), and the goal is to estimate the transition rate between them using the committor function. By applying the algorithm described above, we can obtain the transition rates $k_{AB}$ and $k_{BA}$, as well as the committor function $q_\theta(x)$, which quantifies the probability of transitioning from one basin to the other.


## Installation:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/committor-rate-estimation.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Conclusion:

This algorithm provides a robust and flexible method for estimating transition rates in molecular systems. By combining importance sampling, neural networks, and swarm-based sampling, it is well-suited for modeling complex transition events and estimating rates in biomolecular systems.

## Credits:

* **Algorithm Inspiration**: The approach is inspired by methods used in transition path sampling and forward flux sampling, with significant contributions from the field of molecular dynamics simulations.
* **Müller-Brown Potential**: A classic model in transition state theory.

