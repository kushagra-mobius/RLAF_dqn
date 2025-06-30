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
model = rewardf()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def process_state_and_action(state_dict, update_weights=False, epsilon=0.1):
    # Convert state_dict to ordered list of values
    ordered_state_values = [state_dict[p["key"]] for p in params]
    state_tensor = torch.tensor([ordered_state_values], dtype=torch.float32)

    if random.random() < epsilon and update_weights: # Epsilon-greedy for exploration during training
        action = random.randint(0, len(action_list) - 1)
    else:
        action = get_best_action(state_tensor, model)

    if update_weights:
        reward = calculate_reward(ordered_state_values, action)

        q_values = model(state_tensor)
        q_target = q_values.clone().detach()
        q_target[0, action] = reward

        loss = loss_fn(q_values, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return action

test_state_dict_no_update = {"accuracy": 0.9, "latency": 0.2, "epochs": 8}
best_action_id_no_update = process_state_and_action(test_state_dict_no_update, update_weights=False)
best_action_name_no_update = next((a["name"] for a in action_list if a["id"] == best_action_id_no_update), "Unknown")
print(f"\nTest state (no update): {test_state_dict_no_update}")
print("Best action ID (no update):", best_action_id_no_update)
print("Best action Name (no update):", best_action_name_no_update)

test_state_dict_with_update = {"accuracy": 0.8, "latency": 0.3, "epochs": 12}
best_action_id_with_update = process_state_and_action(test_state_dict_with_update, update_weights=True)
best_action_name_with_update = next((a["name"] for a in action_list if a["id"] == best_action_id_with_update), "Unknown")
print(f"\nTest state (with update): {test_state_dict_with_update}")
print("Best action ID (with update):", best_action_id_with_update)
print("Best action Name (with update):", best_action_name_with_update)


