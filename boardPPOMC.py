import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

BOARD_SIZE = 8
STATE_SIZE = BOARD_SIZE * BOARD_SIZE
N_ACTIONS = 4  # up, down, left, right

def get_state_vector(row, col):
    state = torch.zeros(STATE_SIZE)
    idx = row * BOARD_SIZE + col
    state[idx] = 1.0
    return state

def get_reward(row, col):
    return 100.0 if row == 0 and col == BOARD_SIZE-1 else 0.0

def step(row, col, action):
    if action == 0:   # up
        row = max(0, row - 1)
    elif action == 1: # down
        row = min(BOARD_SIZE - 1, row + 1)
    elif action == 2: # left
        col = max(0, col - 1)
    elif action == 3: # right
        col = min(BOARD_SIZE - 1, col + 1)
    reward = get_reward(row, col)
    done = (row == 0 and col == BOARD_SIZE - 1)
    return row, col, reward, done

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, N_ACTIONS)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

policy_net = PolicyNet()
value_net = ValueNet()
policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)

def run_episode(max_steps=50):
    row, col = 7, 0 # start bottom-left
    states, actions, rewards, positions = [], [], [], [(row, col)]
    for _ in range(max_steps):
        state = get_state_vector(row, col)
        probs = policy_net(state)
        action = torch.multinomial(probs, 1).item()
        states.append(state)
        actions.append(action)
        row, col, reward, done = step(row, col, action)
        rewards.append(reward)
        positions.append((row, col))
        if done:
            break
    return states, actions, rewards, positions

def compute_returns(rewards, gamma=0.98):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def print_board(path):
    grid = [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for (r, c) in path:
        grid[r][c] = 'A'
    grid[0][BOARD_SIZE-1] = 'G'
    for r in range(BOARD_SIZE):
        print(' '.join(grid[r]))
    print()

# ======== Train for several episodes ========
import copy
old_policy_net = copy.deepcopy(policy_net)
eps = 1e-10
for episode in range(1000):
    states, actions, rewards, positions = run_episode()
    returns = compute_returns(rewards)
    if len(returns) == 0:
        continue
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    policy_losses = []
    value_losses = []

    for state, action, G in zip(states, actions, returns):
        value = value_net(state)
        advantage = G - value.item()
        probs = policy_net(state)

        # Surrogate Objective
        with torch.no_grad():
            old_probs = old_policy_net(state)
        ratio = probs[action] / old_probs[action]
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
        policy_loss = -torch.min(surr1, surr2) # min(rA, clip(r)A)

        policy_losses.append(policy_loss)
        value_losses.append(F.mse_loss(value, torch.tensor([G])))

    policy_optimizer.zero_grad()
    torch.stack(policy_losses).sum().backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    torch.stack(value_losses).mean().backward()
    value_optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode:3d}: total_reward={sum(rewards):.1f}, steps={len(rewards)}")
        print_board(positions)
        time.sleep(0.5)

    old_policy_net = copy.deepcopy(policy_net)
