import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025, 
                 gamma=0.95, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.075, 
                 replay_buffer_capacity=100_000, 
                 batch_size=32, 
                 sync_network_rate=10000):
        
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        
        self.x_max_curr = 0 
        self.x_max = 5000 #custom reward function
        
        self.timer_stag = 0
        self.timer_move = 0
                
                
    def get_custom_reward(self, prev_info, curr_info, action, done):
        reward = 0
        max_reward = 5
        rx, ry = .5, .5 
        
        stagnation_constant = 16   #activate stagnation penalty
        move_constant = 30          #activate move reward
        
        prev_x = prev_info.get('x_pos',0)
        curr_x = curr_info.get('x_pos', 0)
        
        prev_y = prev_info.get('y_pos', 0)
        curr_y = curr_info.get('y_pos', 0)
        
        if curr_x>self.x_max_curr:
            
            self.timer_stag = 0
            self.timer_move += 1
            self.x_max_curr = curr_x
            
            if abs(curr_x-prev_x) > 2:
                reward += min(rx*(curr_x - prev_x),max_reward)
                
                if curr_y > prev_y :  # jumping
                    reward += min(ry*(curr_y - prev_y), max_reward)
                if self.timer_move > move_constant:
                    reward += max_reward/4
                    
                    
            
        else:
            self.timer_stag +=1
            self.timer_move = 0

            if self.timer_stag > stagnation_constant:
                reward -= max_reward/30
                self.timer_stag = 0

    
        if curr_info.get("status") == "tall" and prev_info.get("status") != "tall":
            reward += max_reward/2
        elif curr_info.get("status") != "tall" and prev_info.get("status") == "tall":
            reward -= max_reward/4

        if curr_info.get("status") == "fireball" and prev_info.get("status") != "fireball":
            reward += max_reward/2
        elif curr_info.get("status") != "fireball" and prev_info.get("status") == "fireball":
            reward -= max_reward/4

        if curr_info.get("flag_get", False) and not prev_info.get("flag_get", False):
            reward += 3*max_reward  # for beating the level
        elif done:
            reward -= 2*(1-curr_x/self.x_max)*max_reward
            print(f'You lost! Reward: {reward}')        
        return reward

    def choose_action(self, observation):
        # Monte Carlo sampling
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
   
        # LazyFrame wrapper makes observation a list of numpy arrays 
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Highest Q-value index
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        
        # Modify reward based on custom conditions
        prev_state = state  # Previous state (before action)
        curr_state = next_state  # Current state (after action)
                
        
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()  
        self.optimizer.zero_grad()
        
        # Saved to form target network
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]
        
        predicted_q_values = self.online_network(states) # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]


        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # future reward aborted in Bellman eq if DONE flag triggered
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # Double Q-learning step
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

