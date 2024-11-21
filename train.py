import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # SUMO binary checker
import traci  # SUMO TraCI interface


def count_vehicles_on_lanes(lanes):
    """Count the number of vehicles on each lane with a threshold position."""
    vehicle_counts = {lane: 0 for lane in lanes}
    for lane in lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        vehicle_counts[lane] = sum(1 for vid in vehicle_ids if traci.vehicle.getLanePosition(vid) > 10)
    return vehicle_counts


def calculate_total_waiting_time(lanes):
    """Calculate the total waiting time of vehicles on the given lanes."""
    return sum(traci.lane.getWaitingTime(lane) for lane in lanes)


def calculate_reward(junction, lanes):
    """Calculate the reward for the agent's action."""
    waiting_time = calculate_total_waiting_time(lanes)
    vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
    # Penalize low vehicle counts to encourage movement
    movement_penalty = -10 if vehicle_count == 0 else 0
    reward = -waiting_time + movement_penalty
    return reward


def create_traffic_state(junction_id, action):
    """Create a valid traffic light state string for the given junction and action."""
    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
    num_lanes = len(controlled_lanes)

    states = {
        0: "G" * num_lanes,  # All green
        1: "R" * num_lanes,  # All red
        2: "GY" * (num_lanes // 2) + "R" * (num_lanes % 2),  # Alternating green/yellow
        3: "YR" * (num_lanes // 2) + "G" * (num_lanes % 2),  # Alternating yellow/red
    }
    return states.get(action % len(states), "R" * num_lanes)  # Default to all red


def set_traffic_phase(junction_id, duration, state):
    """Set the traffic light phase and duration for a given junction."""
    expected_length = len(traci.trafficlight.getControlledLanes(junction_id))
    if len(state) != expected_length:
        print(f"Debug Info - Junction: {junction_id}, State: {state}, Expected Length: {expected_length}")
        raise ValueError(f"State length {len(state)} does not match expected {expected_length}")
    traci.trafficlight.setRedYellowGreenState(junction_id, state)
    traci.trafficlight.setPhaseDuration(junction_id, duration)


class PPOActorCritic(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, n_actions)
        self.value = nn.Linear(fc2_dims, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return policy, value


class PPOAgent:
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr, gamma, epsilon, batch_size, clip):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.clip = clip
        self.policy = PPOActorCritic(input_dims, fc1_dims, fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = {"states": [], "actions": [], "rewards": [], "log_probs": [], "values": [], "dones": []}

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.memory["states"].append(state)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["log_probs"].append(log_prob)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.policy.device)
        policy, value = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob, value

    def learn(self):
        states = torch.tensor(self.memory["states"], dtype=torch.float).to(self.policy.device)
        actions = torch.tensor(self.memory["actions"]).to(self.policy.device)
        rewards = torch.tensor(self.memory["rewards"], dtype=torch.float).to(self.policy.device)
        log_probs_old = torch.tensor(self.memory["log_probs"]).to(self.policy.device)
        values = torch.tensor(self.memory["values"], dtype=torch.float).to(self.policy.device)
        dones = torch.tensor(self.memory["dones"], dtype=torch.bool).to(self.policy.device)

        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float).to(self.policy.device)

        advantages = returns - values

        for _ in range(10):  # Number of training epochs per update
            policy, values_new = self.policy(states)
            action_dist = torch.distributions.Categorical(policy)
            log_probs_new = action_dist.log_prob(actions)
            ratio = torch.exp(log_probs_new - log_probs_old)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = (returns - values_new).pow(2).mean()
            entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for key in self.memory.keys():
            self.memory[key] = []


def train_traffic_agent(train=True, model_name="traffic_model", net_file="maps/city1.net.xml", epochs=3, steps=500):
    """Train or test the PPO agent for traffic light management."""
    # Select SUMO binary dynamically based on training or testing
    sumo_binary = checkBinary("sumo-gui") if not train else checkBinary("sumo")

    # Extract the base name of the net file to append to the model name
    net_base_name = os.path.splitext(os.path.basename(net_file))[0]
    if train:
        model_path = os.path.join("models", f"{model_name}_{net_base_name}.pth")
    else:
        # Use the provided model_name as the exact model path during testing
        model_path = model_name

    if not os.path.exists("models"):
        os.makedirs("models")

    agent = PPOAgent(
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        n_actions=4,
        lr=0.0003,
        gamma=0.99,
        epsilon=0.2,
        batch_size=64,
        clip=0.2,
    )

    if not train:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            agent.policy.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    traci.start([sumo_binary, "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])
    junctions = traci.trafficlight.getIDList()

    total_times = []
    for epoch in range(epochs):
        total_waiting_time = 0
        for step in range(steps):
            traci.simulationStep()
            for junction in junctions:
                lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = calculate_total_waiting_time(lanes)
                total_waiting_time += waiting_time

                state = list(count_vehicles_on_lanes(lanes).values())
                action, log_prob, value = agent.choose_action(state)

                # Apply action
                state_str = create_traffic_state(junction, action)
                reward = calculate_reward(junction, lanes)
                set_traffic_phase(junction, duration=6, state=state_str)
                done = step == steps - 1
                agent.store_transition(state, action, reward, log_prob, value, done)

            if train and (step % agent.batch_size == 0 or step == steps - 1):
                agent.learn()

        total_times.append(total_waiting_time)
        print(f"Epoch {epoch + 1}/{epochs}, Total Waiting Time: {total_waiting_time}")
        if not train:
            break

    traci.close()

    if train:
        # Save the model with a unique name for the net file
        torch.save(agent.policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        plt.plot(total_times)
        plt.title("Total Waiting Time per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Total Waiting Time")
        plt.show()


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Train or test the PPO agent for SUMO traffic light management.")
    parser.add_argument(
        "--train",
        type=lambda x: x.lower() == "true",  # Convert "true"/"false" strings to boolean
        default=True,
        help="Specify whether to train (True) or test (False) the model.",
    )
    parser.add_argument(
        "--net_file",
        type=str,
        default="maps/city1.net.xml",
        help="Path to the SUMO .net.xml file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="traffic_model",
        help="Base name for the saved model file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to run during training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the traffic agent
    train_traffic_agent(
        train=args.train,
        model_name=args.model_name,
        net_file=args.net_file,
        epochs=args.epochs,
        steps=args.steps,
    )


# To train: python train.py --train true --net_file "maps/city4.net.xml" --steps 500
# To test: python train.py --train false --net_file "maps/city4.net.xml" --model_name "models/traffic_model_city3.net.pth" --epochs 1 --steps 500