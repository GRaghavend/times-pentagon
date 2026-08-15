import gymnasium as gym
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#Hyperparameters
with open('hyperparameters.yml', 'r') as f:
    config = yaml.safe_load(f)

GAMMA = config['cartpole1']['gamma']
LEARNING_RATE = config['cartpole1']['learning_rate']
HIDDEN_LAYER = config['cartpole1']['hidden_layer']
MAX_EPISODES = config['cartpole1']['max_episodes']
MAX_STEPS = config['cartpole1']['max_steps']
ENV_NAME = config['cartpole1']['env_name']
TEST_EPISODES = config['cartpole1']['test_episodes']

#Actor Critic Network
class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.actor = nn.Linear(hidden_size,action_dim)
        
        # A single scalar value that says 
        # how much total future reward do I expect from the state, 
        # playing my current policy
        self.critic = nn.Linear(hidden_size,1) 
        

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x),dim=1)
        state_value = self.critic(x)
        return action_probs,state_value

def test_A2C(model,episodes=5,render=False):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)

    for episode in range(episodes):
        state , _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0

        done = False

        while not done:
            with torch.no_grad(): #Since we are not training we don't want to update the parameters
                action_probs, _ = model(state)
                action = torch.argmax(action_probs, dim=1).item()

            next_state,reward,terminated,truncated,_ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            state = torch.FloatTensor(next_state).unsqueeze(0)

        print(f"Test Episodes: {episode + 1}: Total Reward = {total_reward}")

    env.close()
        
#Training Loop
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

#model definition
model = ActorCritic(state_dim,action_dim,HIDDEN_LAYER)
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

# Run the loop
episode_rewards = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    episode_reward = 0
    log_probs, values, rewards = [],[],[]

    for step in range(MAX_STEPS):
        action_probs, value = model(state)
        dist = Categorical(action_probs) # converts the probs into distribution
        action = dist.sample()
        log_prob = dist.log_prob(action) #Convert the action probability into log probability 

        next_state, reward, terminated, truncated , _ = env.step(action.item())
        done = terminated or truncated

        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        episode_reward += reward 

        log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(reward)

        state = next_state

        if done:
            break
    returns = []

    R = 0

    #weightage function. Gives more weightage to immediate rewards. 

    for r in rewards[::-1]:
        R = r + GAMMA*R
        returns.insert(0,R)

    returns = torch.FloatTensor(returns)

    #Normalize the returns
    returns = (returns - returns.mean())/(returns.std() + 1e-8)

    actor_loss, critic_loss = 0,0

    for log_prob, value, R in zip(log_probs,values,returns):
        advantage = R - value.item()
        actor_loss += -log_prob * advantage
        critic_loss += (R-value)**2

    #Normalize the values
    actor_loss /= len(log_probs)
    critic_loss /= len(values)

    optimizer.zero_grad()
    total_loss = actor_loss + critic_loss

    total_loss.backward()
    optimizer.step()

    episode_rewards.append(episode_reward)

    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode}, Avg Reward {avg_reward}")

print(f"Train complete")

test_A2C(model, episodes=TEST_EPISODES)