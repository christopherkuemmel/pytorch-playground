import numpy as np
from game import Game

import torch
import torch.nn as nn
import torch.optim as optim

# define actions
actions = ('left', 'down', 'right', 'up')


def print_policy():
    policy = [agent(s).argmax(1)[0].detach().item() for s in range(state_space)]
    policy = np.asarray([actions[action] for action in policy])
    policy = policy.reshape((game.max_row, game.max_col))
    print("\n\n".join('\t'.join(line) for line in policy)+"\n")


# define Q-Network
class QNetwork(nn.Module):

    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        # simple linear one-layer feed-forward
        self.linear1 = nn.Linear(state_space, action_space)
        self.state_space = state_space

    def forward(self, x):
        x = self.one_hot_encoding(x)
        return self.linear1(x)

    def one_hot_encoding(self, x):
        '''
        One-hot encodes the input data, based on the defined state_space. 
        '''
        out_tensor = torch.zeros([1, state_space])
        out_tensor[0][x] = 1
        return out_tensor


# Make use of cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Game Instance
game = Game(living_penalty=-0.04, render=False)

# Define State and Action Space
state_space = game.max_row * game.max_col
action_space = len(actions)

# Set learning parameters
e = 0.1  # epsilon
lr = .03  # learning rate
y = .999  # discount factor
num_episodes = 2000

# create lists to contain total rewards and steps per episode
jList = []
rList = []

# init Q-Network
agent = QNetwork(state_space, action_space).to(device)

# define optimizer and loss
optimizer = optim.Adam(agent.parameters(), lr=lr) # .SGD(agent.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = game.reset()
    rAll = 0
    j = 0

    # The Q-Network learning algorithm
    while j < 99:
        j += 1

        # Choose an action by greedily (with e chance of random action) from the Q-network
        with torch.no_grad():
            a = agent(s).max(1)[1].view(1, 1)
        if np.random.rand(1) < e:
            a[0][0] = np.random.randint(1, 4)

        # Get new state and reward from environment
        r, s1, game_over = game.perform_action(actions[a.item()])

        # Calculate Q and target Q
        q = agent(s).max(1)[0].view(1, 1)
        target_q = r + y * agent(s1).max(1)[0].view(1, 1).detach()

        # Calculate loss
        loss = criterion(q, target_q)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add reward to list
        rAll += r

        # Replace old state with new
        s = s1

        if game_over:
            # Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    rList.append(rAll)
    jList.append(j)

print("\Average steps per episode: " + str(sum(jList)/num_episodes))
print("\nScore over time: " + str(sum(rList)/num_episodes))
print("\nFinal Q-Network Policy:\n")
print_policy()
