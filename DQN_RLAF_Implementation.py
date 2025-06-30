import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class rewardf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 4)  # 4 actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_user_defined_reward_function():
    params = {}

    for i in range(1, 4):
        pname = input(f"Enter name of parameter {i} (e.g., accuracy, latency, epochs): ").strip()
        sign = input(f"Should {pname} be '+' or '-'?: ").strip()
        mul = float(input(f"What should {pname} be multiplied by?: ").strip())

        params[pname] = (sign, mul)

    
    def reward_function(accuracy, latency, epochs, action):
        
        p_values = {"accuracy": accuracy, "latency": latency, "epochs": epochs}
        reward = 0.0
        for key, (sign, mul) in params.items():
            if sign == '+':
                reward += p_values[key] * mul
            elif sign == '-':
                reward -= p_values[key] * mul
            else:
                raise ValueError(f"Invalid sign '{sign}' for parameter '{key}'")

        if action == 0:
            reward *= 1.0
        elif action == 1:
            reward *= 0.8
        elif action == 2:
            reward *= 1.3
        elif action == 3:
            reward *= 1.5

        return reward

    return reward_function

episodes = 1000
epsilon = 0.1
lr = 0.01
model = rewardf()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


print("\n Define your reward function using 3 parameters (accuracy, latency, epochs):")
reward_fn = get_user_defined_reward_function()

for episode in range(episodes):
    accuracy = random.uniform(0.1, 1.0)
    latency = random.uniform(0.1, 2.0)
    epochs_val = random.randint(1, 20)
    state = torch.tensor([[accuracy, latency, epochs_val]], dtype=torch.float32)

    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

    reward = reward_fn(accuracy, latency, epochs_val, action)

    q_values = model(state)
    q_target = q_values.clone().detach()
    q_target[0, action] = reward

    loss = loss_fn(q_values, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_input = torch.tensor([[0.9, 0.2, 8]], dtype=torch.float32)
with torch.no_grad():
    predicted_qs = model(test_input)
    best_action = torch.argmax(predicted_qs).item()
    print("\n Test state: [accuracy=0.9, latency=0.2, epochs=8]")
    print("Predicted Q-values:", predicted_qs.numpy())
    print("Best action:", best_action)
