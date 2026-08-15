import gymnasium
import flappy_bird_gymnasium

import torch
from torch import nn

from dqn import DQN
from experience_replay import ReplayMemory

import itertools
import yaml
import random
import os
import argparse
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta



#For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

#Directory for saving the run
RUNS_DIR = "DQN/runs"
os.makedirs(RUNS_DIR,exist_ok=True)

#Used to generate plots and save the file instead of rendering all the time
matplotlib.use('Agg')

if torch.cuda.is_available():
    device = "cuda"
#elif torch.backends.mps.is_available():
    #device = "mps"
else:
    device = "cpu"
print(device)

class Agent:
    def __init__(self,hyperparameter_set):
        
        with open("DQN/hyperparameters.yml",'r') as file:
            all_hyperparameters_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_set[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{}) 
        
        
        #Neural Network hyperparameters
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        #Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameter_set}.png')
        
    def run(self,is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started for hyperparameter set '{self.hyperparameter_set}'."
            print(log_message)
            
            with open(self.LOG_FILE,'w') as file:
                file.write(log_message + '\n')
                
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, **self.env_make_params)
        # env = gymnasium.make("CartPole-v1",render_mode="human" if render else None)
        
        num_state = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []
        
        policy_dqn = DQN(num_state,num_actions,self.fc1_nodes).to(device)
        
        if is_training:
            #Initialize epsilon
            epsilon = self.epsilon_init
            
            memory = ReplayMemory(self.replay_memory_size)
            
            #Target DQN
            target_dqn = DQN(num_state,num_actions,self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict()) #Copies weights and biases of the policy network
            
            #Policy network optimizer "Adam" optimizer.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
              
            #List to keep track of epsilons
            epsilon_history = []
            
            #Track the best reward
            best_reward = -999999
                      
            #To track the number of steps taken. Used to sync the policy => target network
            step_count = 0
        else:
            # This is used to load the saved policy from the model file
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE)) 
            
            #switch model to evaluation mode
            policy_dqn.eval()
        
        try:
            #Training loop (Ctrl+C to stop)
            #for episode in itertools.count():
            for episode in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state,dtype=torch.float, device=device)

                terminated = False
                episode_reward = 0.0

                while(not terminated and episode_reward < self.stop_on_reward):
                    # Next action:
                    # (feed the observation to your agent here)
                    # random.random() is used to randomly decide whethere
                    # action is supposed to be explored or exploited
                    # if ε = 0.1, 10 percent random action
                    # 90 percent use the policy given by DQN

                    #----------------------------------#
                    # BLOCK OF CODE TO CHOOSE ACTIONS
                    #----------------------------------#

                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample() #explore
                        action = torch.tensor(action,dtype=torch.int64,device=device)
                    else:
                        with torch.no_grad():
                            # tensor([1,2,3])
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit


                    # Processing:
                    #We choose an action from the above and feed it to the environment
                    #So, we get a new state
                    new_state, reward, terminated, _, info = env.step(action.item())

                    episode_reward += reward

                    #Convert new state to tensor and reward to tensors
                    new_state = torch.tensor(new_state,dtype=torch.float,device=device)
                    reward = torch.tensor(reward,dtype=torch.float,device=device)

                    if is_training:
                        memory.append((state,action,new_state,reward,terminated))
                        step_count += 1

                    #Move to a new state
                    state = new_state

                rewards_per_episode.append(episode_reward)

                if is_training:
                    #Epsilon alteration
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if episode_reward > best_reward:
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} at episode {episode} (previous best: {best_reward:0.1f}). Saving model."
                        print(log_message)

                        with open(self.LOG_FILE,'a') as file:
                            file.write(log_message + '\n')

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                        current_time = datetime.now()
                        if current_time - last_graph_update_time > timedelta(seconds=20):
                            self.save_graph(rewards_per_episode,epsilon_history)
                            last_graph_update_time = current_time

                    #to check for enough experience
                    if len(memory)>self.mini_batch_size:
                        #Sample from memory
                        mini_batch = memory.sample(self.mini_batch_size)

                        self.optimize(mini_batch,policy_dqn,target_dqn)

                        #Copy policy network to target network after a certain number of steps
                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0
        finally:
            if is_training and rewards_per_episode:
                self.save_graph(rewards_per_episode, epsilon_history)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # For each stored experience: compute what the Q-value should have been (target_q, using the reward plus discounted future value from target_dqn), 
    # compare it to what policy_dqn currently predicts (current_q), and nudge the network's weights to shrink that gap.
    # A more easier version to understand
    '''
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        
        for state,action,new_state,reward,terminated in mini_batch:
            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            
            current_q = policy_dqn(state)
            
            #Compute loss for the whole minibatch
            loss = self.loss_fn(current_q,target_q)
            
            #Optimize the model
            self.optimizer.zero_grad() # Clear gradients
            loss.backward() # Compute gradients (backprop) - slope 
            self.optimizer.step() #Update the weights and biases 
        '''
        
if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
    