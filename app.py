import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    env = gym.make("Acrobot-v1")

    n_actions = env.action_space.n
    observation_size = env.observation_space.shape[0]

    print("Observation size:", observation_size)
    print("Number of actions:", n_actions)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # DQN hyperparameters
    hidden_size = 128
    learning_rate = 0.001
    gamma = 0.99

    replay_buffer_size = 50_000
    minimum_replay_size = 1_000
    batch_size = 64
    target_update_frequency = 500

    initial_epsilon = 1.0
    minimum_epsilon = 0.05
    epsilon_decay = 0.995
    epsilon = initial_epsilon

    HORIZON = 500
    MAX_TRAJECTORIES = 1000

    # The online network is used to select actions and is updated during training
    online_network = create_q_network(
        observation_size=observation_size,
        hidden_size=hidden_size,
        n_actions=n_actions
    ).to(device)

    # The target network is used to calculate stable DQN target values
    target_network = create_q_network(
        observation_size=observation_size,
        hidden_size=hidden_size,
        n_actions=n_actions
    ).to(device)

    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(online_network.parameters(), lr=learning_rate)

    loss_function = nn.SmoothL1Loss()

    replay_buffer = deque(maxlen=replay_buffer_size)

    scores = []
    rewards = []
    losses = []
    td_errors = []
    successes = []
    epsilon_history = []

    total_training_steps = 0
    rng = np.random.default_rng()

    training_start_time = time.perf_counter()

    for trajectory in range(MAX_TRAJECTORIES):
        current_observation, info = env.reset()

        current_observation = np.asarray(
            current_observation,
            dtype=np.float32
        )

        total_reward = 0.0
        total_episode_loss = 0.0
        total_absolute_td_error = 0.0
        training_updates = 0
        step_count = 0

        terminated = False
        truncated = False

        for step in range(HORIZON):
            # Step 1: Select an action using epsilon-greedy exploration
            action = select_action(
                q_network=online_network,
                observation=current_observation,
                epsilon=epsilon,
                n_actions=n_actions,
                rng=rng,
                device=device
            )

            # Step 2: Execute the action
            (next_observation, reward, terminated, truncated, info) = env.step(action)

            next_observation = np.asarray(next_observation, dtype=np.float32)

            done = terminated or truncated

            # Step 3: Store the transition in experience replay memory
            replay_buffer.append(
                (
                    current_observation,
                    action,
                    float(reward),
                    next_observation,
                    float(terminated)
                )
            )

            total_reward += reward
            step_count += 1
            total_training_steps += 1

            # Step 4: Train the online network using a random mini-batch
            if len(replay_buffer) >= minimum_replay_size:
                loss_value, average_absolute_td_error = train_dqn_batch(
                    online_network=online_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device
                )

                total_episode_loss += loss_value
                total_absolute_td_error += average_absolute_td_error
                training_updates += 1

            # Step 5: Periodically synchronize the target network
            if total_training_steps % target_update_frequency == 0:
                target_network.load_state_dict(
                    online_network.state_dict()
                )

            if done:
                break

            current_observation = next_observation

        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

        scores.append(step_count)
        rewards.append(total_reward)
        epsilon_history.append(epsilon)

        if training_updates > 0:
            average_episode_loss = (
                total_episode_loss / training_updates
            )

            average_episode_td_error = (
                total_absolute_td_error / training_updates
            )
        else:
            average_episode_loss = 0.0
            average_episode_td_error = 0.0

        losses.append(average_episode_loss)
        td_errors.append(average_episode_td_error)

        # Acrobot terminates successfully when its free end reaches the required height. A time-limit truncation is not counted as success
        success = terminated and not truncated
        successes.append(success)

        completed_trajectories = trajectory + 1

        if completed_trajectories % 50 == 0:
            average_score = np.mean(scores[-50:])
            average_reward = np.mean(rewards[-50:])
            success_rate = np.mean(successes[-50:]) * 100
            average_loss = np.mean(losses[-50:])
            average_td_error = np.mean(td_errors[-50:])

            elapsed_time = (
                time.perf_counter()
                - training_start_time
            )

            print(
                f"Trajectory {completed_trajectories}\t"
                f"Average Score: {average_score:.2f}\t"
                f"Average Reward: {average_reward:.2f}\t"
                f"Success Rate: {success_rate:.2f}%\t"
                f"Average Loss: {average_loss:.4f}\t"
                f"Average TD Error: {average_td_error:.4f}\t"
                f"Epsilon: {epsilon:.4f}\t"
                f"Replay Size: {len(replay_buffer)}\t"
                f"Training Time: {elapsed_time:.2f}s"
            )

    training_time = (
        time.perf_counter()
        - training_start_time
    )

    env.close()

    print("\nTraining complete.")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Final 50-trajectory success rate: {np.mean(successes[-50:]) * 100:.2f}%")
    print(f"Final 50-trajectory average score: {np.mean(scores[-50:]):.2f}")
    print(f"Final 50-trajectory average loss: {np.mean(losses[-50:]):.4f}")

    score_array = np.array(scores)

    # Tracking four performance-related metrics:
    # 1. Number of steps per trajectory
    # 2. Success rate
    # 3. DQN training loss
    # 4. Average absolute TD error
    generate_scatter_plot(score_array)
    generate_success_rate_plot(np.array(successes))
    generate_loss_plot(np.array(losses))
    generate_td_error_plot(np.array(td_errors))


def create_q_network(observation_size, hidden_size, n_actions):
    # The network receives the six continuous Acrobot observation values and outputs one estimated Q-value for each possible action
    return nn.Sequential(
        nn.Linear(observation_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions)
    )


def select_action(q_network, observation, epsilon, n_actions, rng, device):
    # Exploration: choose a random action
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    # Exploitation: choose the action with the highest predicted Q-value
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = q_network(observation_tensor)[0]

    maximum_q_value = torch.max(q_values)

    best_actions = torch.nonzero(torch.isclose(q_values, maximum_q_value), as_tuple=False).squeeze(1)

    selected_index = int(rng.integers(len(best_actions)))

    return int(best_actions[selected_index].item())


def train_dqn_batch(online_network, target_network, optimizer, loss_function, replay_buffer, batch_size, gamma, device):
    transitions = random.sample(replay_buffer, batch_size)

    (observations, actions, rewards, next_observations, terminated_values) = zip(*transitions)

    observation_tensor = torch.as_tensor(
        np.array(observations),
        dtype=torch.float32,
        device=device
    )

    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.int64,
        device=device
    ).unsqueeze(1)

    reward_tensor = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_observation_tensor = torch.as_tensor(
        np.array(next_observations),
        dtype=torch.float32,
        device=device
    )

    terminated_tensor = torch.as_tensor(
        terminated_values,
        dtype=torch.float32,
        device=device
    )

    # Obtain Q(s, a) for each action selected in the sampled transitions
    current_q_values = online_network(
        observation_tensor
    ).gather(
        1,
        action_tensor
    ).squeeze(1)

    # DQN target:
    # target = reward for terminal states
    # target = reward + gamma * max Q(s', a') otherwise
    with torch.no_grad():
        best_next_q_values = target_network(
            next_observation_tensor
        ).max(dim=1).values

        target_q_values = (
            reward_tensor
            + gamma
            * (1.0 - terminated_tensor)
            * best_next_q_values
        )

    td_errors = target_q_values - current_q_values

    loss = loss_function(
        current_q_values,
        target_q_values
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_norm=10.0
    )

    optimizer.step()

    return (
        float(loss.item()),
        float(td_errors.abs().mean().item())
    )


def generate_scatter_plot(score):
    average_score = running_mean(score)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Trajectory Duration", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    plt.plot(
        np.arange(len(score)),
        score,
        color="gray",
        linewidth=1,
        label="Trajectory duration"
    )

    plt.scatter(
        np.arange(len(score)),
        score,
        color="green",
        linewidth=0.3,
        label="Individual trajectory"
    )

    if len(average_score) > 0:
        average_x = np.arange(
            49,
            49 + len(average_score)
        )

        plt.plot(
            average_x,
            average_score,
            color="blue",
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Deep Q-Network Performance on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("\nClose the trajectory plot to view the next plot.")
    plt.show()


def generate_success_rate_plot(successes, window_size=50):
    success_percentages = successes.astype(float) * 100

    rolling_success_rate = running_mean(
        success_percentages,
        window_size
    )

    plt.figure(figsize=(15, 7))
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    if len(rolling_success_rate) > 0:
        success_x = np.arange(
            window_size - 1,
            window_size - 1 + len(rolling_success_rate)
        )

        plt.plot(
            success_x,
            rolling_success_rate,
            linewidth=3,
            label=f"{window_size}-trajectory success rate"
        )

    plt.ylim(-5, 105)
    plt.title("Deep Q-Network Success Rate on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("Close the success-rate plot to view the next plot.")
    plt.show()


def generate_loss_plot(losses):
    average_loss = running_mean(losses)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Average DQN Loss", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    plt.plot(
        np.arange(len(losses)),
        losses,
        linewidth=1,
        label="Trajectory loss"
    )

    if len(average_loss) > 0:
        average_x = np.arange(
            49,
            49 + len(average_loss)
        )

        plt.plot(
            average_x,
            average_loss,
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Deep Q-Network Loss on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("Close the loss plot to view the next plot.")
    plt.show()


def generate_td_error_plot(td_errors):
    average_td_error = running_mean(td_errors)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Average Absolute TD Error", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    plt.plot(
        np.arange(len(td_errors)),
        td_errors,
        linewidth=1,
        label="Trajectory TD error"
    )

    if len(average_td_error) > 0:
        average_x = np.arange(
            49,
            49 + len(average_td_error)
        )

        plt.plot(
            average_x,
            average_td_error,
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Deep Q-Network TD Error on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("Run complete. Close the plot window to exit.")
    plt.show()


def running_mean(values, window_size=50):
    if len(values) < window_size:
        return np.array([])

    kernel = np.ones(window_size) / window_size

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )


if __name__ == "__main__":
    main()
