import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


params = [
    {"key": "accuracy", "sign": "+", "mul": 1.0},
    {"key": "latency", "sign": "-", "mul": 0.5},
    {"key": "epochs", "sign": "-", "mul": 0.1}
]

action_list = [
    {"id": 0, "name": "Action_A"},
    {"id": 1, "name": "Action_B"},
    {"id": 2, "name": "Action_C"},
    {"id": 3, "name": "Action_D"}
]

class rewardf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(params), 16)
        self.fc2 = nn.Linear(16, len(action_list))
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def calculate_reward(state_values, action_id):
    reward = 0.0
    p_values = {p["key"]: val for p, val in zip(params, state_values)}

    for p in params:
        key = p["key"]
        sign = p["sign"]
        mul = p["mul"]
        
        if sign == '+':
            reward += p_values[key] * mul
        elif sign == '-':
            reward -= p_values[key] * mul
        else:
            raise ValueError(f"Invalid sign '{sign}' for parameter '{key}'")

    return reward

def get_best_action(state, model, print_q_values=True):
    with torch.no_grad():
        q_values = model(state)
        if print_q_values:
            probabilities = F.softmax(q_values, dim=1)
            print("Action 'Probabilities' (Softmax of Q-values):", probabilities.numpy())
        best_action_id = torch.argmax(q_values).item()
        return best_action_id

lr = 0.01
gamma = 0.99  # Discount factor
TARGET_UPDATE_FREQ = 10  # How often to update the target network

policy_net = rewardf()
target_net = rewardf()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # Target network is not trained directly

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def get_best_action(state, model, epsilon=0.0, print_q_values=True):
    if random.random() < epsilon:
        return random.randint(0, len(action_list) - 1)
    else:
        with torch.no_grad():
            q_values = model(state)
            if print_q_values:
                probabilities = F.softmax(q_values, dim=1)
                print("Action 'Probabilities' (Softmax of Q-values):", probabilities.numpy())
            best_action_id = torch.argmax(q_values).item()
            return best_action_id

def train_dqn_step(prev_state_dict, action_taken, new_state_dict):
    # Convert state_dicts to ordered list of values
    prev_ordered_state_values = [prev_state_dict[p["key"]] for p in params]
    prev_state_tensor = torch.tensor([prev_ordered_state_values], dtype=torch.float32)

    new_ordered_state_values = [new_state_dict[p["key"]] for p in params]
    new_state_tensor = torch.tensor([new_ordered_state_values], dtype=torch.float32)

    reward = calculate_reward(new_ordered_state_values, action_taken)

    # Get Q-values for the previous state from the policy network
    q_values = policy_net(prev_state_tensor)

    # Double DQN: Select best action from policy_net for the new_state
    # and then evaluate it using the target_net
    with torch.no_grad():
        next_q_values_policy = policy_net(new_state_tensor)
        best_next_action = torch.argmax(next_q_values_policy, dim=1).item()
        next_q_value_target = target_net(new_state_tensor)[0, best_next_action]

    # Compute the target Q-value
    q_target = q_values.clone().detach()
    q_target[0, action_taken] = reward + gamma * next_q_value_target

    # Compute loss and optimize
    loss = loss_fn(q_values, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Example training loop
num_episodes = 50
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

print("\nStarting Double DQN Training Loop...")
for episode in range(num_episodes):
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

    # Simulate an initial state (for simplicity, using a random state)
    prev_state_dict = {p["key"]: random.uniform(0, 1) for p in params}
    prev_state_dict["epochs"] = random.randint(1, 20) # epochs might be integer

    # Agent selects an action
    prev_state_tensor = torch.tensor([[prev_state_dict[p["key"]] for p in params]], dtype=torch.float32)
    action_taken = get_best_action(prev_state_tensor, policy_net, epsilon=epsilon, print_q_values=False)

    # Simulate environment transition (for simplicity, generating a new random state)
    new_state_dict = {p["key"]: random.uniform(0, 1) for p in params}
    new_state_dict["epochs"] = random.randint(1, 20)

    # Perform a training step
    train_dqn_step(prev_state_dict, action_taken, new_state_dict)

    # Update the target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}: Target network updated.")

    if episode % 10 == 0:
        print(f"Episode {episode}, Epsilon: {epsilon:.4f}")

print("\nDouble DQN Training Complete.")

# Test the trained policy network
final_test_state_dict = {"accuracy": 0.95, "latency": 0.1, "epochs": 5}
final_test_state_tensor = torch.tensor([[final_test_state_dict[p["key"]] for p in params]], dtype=torch.float32)
final_best_action_id = get_best_action(final_test_state_tensor, policy_net, epsilon=0.0, print_q_values=True)
final_best_action_name = next((a["name"] for a in action_list if a["id"] == final_best_action_id), "Unknown")

print(f"\nFinal Test State: {final_test_state_dict}")
print("Final Best Action ID:", final_best_action_id)
print("Final Best Action Name:", final_best_action_name)


