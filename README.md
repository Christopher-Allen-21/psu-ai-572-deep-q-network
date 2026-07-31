# PSU AI-572 Deep Q-Network (DQN) Algorthm

## Tasks for Programming Assignment 3
In this programming assignment you are required to complete the following tasks:
- Implement the DQN RF algorithm to obtain the goal for Acrobot using the Python language.
- Determine all the necessary performance related metrics that you plan to use to evaluate the performance of the RF algorithm.
- Discuss your findings here: Out of the three explored RF algorithms for the Acrobot environment, which algorithm demonstrated a superior performance (better rewards)?

## Overview
Within the world of Reinforcement Learning (RF), OpenAI’s Gymnasium fork of the Gym library provides a standard API to work between learning algorithms and environments. This allows for the use of pre-built environments to be tested with various RF algorithms without the overhead of building said environments. A number of classic control environments exist within the Gymnasium library. These classic control environments are stochastic in regard to the initial state within some given range.

One such control problem is Acrobot. The goal of Acrobot, which simulates the chaotic physics of a double physical pendulum, is to apply a torque to the joint connecting the two bars of the physical pendulum, the actuated joint, and swing the free end of the pendulum up to a given height in as few moves as possible. The process of implementing RF algorithm to obtain the end goal for Acrobot is discussed below:

## The Acrobot Environment
<img width="145" height="165" alt="image" src="https://github.com/user-attachments/assets/0a3282cd-60fc-45e0-89f5-95f3757f9862" />

The Acrobot control task agent is a double physical pendulum. The first pendulum link has one end pinned in space on a pivot while the other end is linked through another pivoting joint to the second free end of the pendulum. The pivoting joint connecting the fixed and free pendulum links is known as the actuating joint. This joint will have torques applied to it to actuate the double pendulum into motion.

The Acrobot system uses a discrete, deterministic action space representing the torque applied to the actuated joint. The action taken can be applying -1, 0, or +1 torque to the actuated joint.

The entire observation state space within Acrobot is made up of six observable features of the double physical pendulum. Three of the features correspond to 𝜃1, the angle of the first fixed joint, while the second set of features correspond to 𝜃2 which is angle of the second link relative to the angle of the first link. For example, consider if the angle of the first link, 𝜃1, was at 45 degrees from vertical. If the second link was in-line with the first link, then 𝜃2 would be equal to zero degrees. For both 𝜃1 and 𝜃2, the observable features are cosine, sine, and angular velocity. Cosine and sine are naturally bounded between -1 and +1. The angular velocity of 𝜃1 and 𝜃2 range from ±4𝜋 rads/sec and ±9𝜋 rads/sec respectively. The Acrobot environments holds this observation space as a "ndarray" of shape (6). A sample state of both physical links pointing down at an unknown angular velocity would be presented as: [1, 0, 1, 0, 𝝎𝜃1, 𝝎𝜃2].

Upon initialization, each parameter within the state is generated uniformly between -0.1 and +0.1. Generally, this will result in both links pointing downwards with some initial randomness to the system. The target height is calculated by the following: − cos(𝜃1) − cos(𝜃2 + 𝜃1) > 1.0.  Termination of the episode can also occur when the episode length is greater than 500.

The goal of Acrobot is to reach the target height in as few steps as possible. Each step of the system that does not reach the target height provides a reward of -1.  A successful swing that reaches the target height results in termination of the episode with a reward of zero. This makes the lowest return -500, a failure to achieve the target height, and the maximum return -1 if the system miraculously reaches the target height in the first step.

## To run locally
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
  
- python /Users/christopherallen/Desktop/Projects/psu-ai-572-deep-q-network/app.py

## Execution/Visualization
- <img width="1346" height="776" alt="Screenshot 2026-07-31 at 9 11 24 AM" src="https://github.com/user-attachments/assets/9d26ccab-01e5-4a97-94ec-32609a50dc2d" />
- <img width="1499" height="757" alt="Screenshot 2026-07-29 at 7 24 02 PM" src="https://github.com/user-attachments/assets/f6dbe9b3-d179-431e-ae6a-378e006f74d2" />
- <img width="1495" height="758" alt="Screenshot 2026-07-29 at 7 24 12 PM" src="https://github.com/user-attachments/assets/b8a42573-e94a-43ce-bdc7-20e0ff1f6e37" />
- <img width="1495" height="761" alt="Screenshot 2026-07-29 at 7 24 15 PM" src="https://github.com/user-attachments/assets/d0774a14-f489-4039-bfa8-2ddc4719d89f" />
- <img width="1499" height="764" alt="Screenshot 2026-07-29 at 7 24 28 PM" src="https://github.com/user-attachments/assets/a3b93143-6a5b-4fa0-ba8f-feeb45563035" />













