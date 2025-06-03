# Introduction to MuJoCo

MuJoCo is a free and open-source physics engine designed to facilitate research and development in robotics, biomechanics, graphics, and animation, providing fast and accurate simulation capabilities.

* **MuJoCo**: **Mu**lti-**Jo**int dynamics with **Co**ntact
* Rigid-body simulation
* Essential simulator components:

  * Solving equations of motion (e.g., articulated body algorithms)
  * Contact solver (key factor for sim-to-real gaps)
  * User-friendly visualizer

## Tutorial Topics

1. Introduction to `MuJoCo` and `mujoco_parser`
2. Forward Kinematics
3. Forward Dynamics
4. Inverse Kinematics
5. Inverse Dynamics
6. Reinforcement Learning with `Soft Actor-Critic` for `Snapbot`

## Environment Setup

Follow the steps below to configure your environment

### 1. Create and Activate Conda Environment

Create a new conda environment and activate it:

```bash
conda create -n snapbot-env python=3.10
conda activate snapbot-env
```

### 2. Install Dependencies
Install all required packages from the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

---

## Project: Snapbot Olympics

### Overview

Optimize robotic motion using reinforcement learning. Train and evaluate control policies in simulation for various competition tasks.

### Events

* **Standing Long Jump:** Forward distance achieved by robot after jumping
* **High Jump:** Vertical height achieved by robot
* **Running Sideways:** Time to reach designated goal coordinate (faster is better)

### Simulation

* All tasks conducted in MuJoCo

### Provided Resources

* Snapbot robot model (`XML`)
* SAC algorithm implementation (`notebook/06_sac_snapbot_train.ipynb`, `notebook/06_sac_snapbot_eval.ipynb`, `pakage/gym/snapbot_env.py`, `pakage/rl/sac.py`)
* *(Reward functions must be designed by students)*

### Tasks

* Configure neural network dimensions based on robot DoF and action dimensions
* Design task-specific reward functions
* Train and evaluate policies
* Analyze reward function impact on performance

### Submission

* One-page A4 result report (must include reward definitions and model hyperparameters)
* Video of trained Snapbot performance

Aim to design the best policy and win the Snapbot Olympics!

---

### Contact

* **Prof. Sungjoon Choi:** [sungjoon dash choi at korea dot ac dot kr](mailto:sungjoon-choi@korea.ac.kr)
* **TA Taemoon Jeong:** [taemoon dash jeong at korea dot ac dot kr](mailto:taemoon-jeong@korea.ac.kr)
