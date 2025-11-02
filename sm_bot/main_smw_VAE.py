from pynput import keyboard
import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

from agent_vae import AgentVAE

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

from human_data_collect import HumanDataCollect
# COMPLEX_MOVEMENT combos include: no-op, left, right, left+A, right+A, A, B, left+B, right+B, A+B, etc.

KEY_TO_BUTTON = {
    'a': 'left',
    'd': 'right',
    'w': 'up',
    's': 'down',
    'j': 'B',  # Jump
    'h': 'A',  # Run
    'k': 'select',  # extra buttons if needed
    'l': 'start',
}

pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def get_action_from_keys():
    buttons_pressed = set()
    for key in pressed_keys:
        btn = KEY_TO_BUTTON.get(key)
        if btn:
            buttons_pressed.add(btn)
    for i, combo in enumerate(COMPLEX_MOVEMENT):
        if buttons_pressed == set(combo):
            return i
    return 0  # NOOP if no combo matches

##############################################################################################
##############################################################################################
####Data Collection/me playing and recording my actions

DATA_COLLECT = False
ENV_NAME = 'SuperMarioBros-1-1-v0'

NUM_TRIALS = 6
DISPLAY = True
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)

#env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, COMPLEX_MOVEMENT)   #hopefully we can find a latent representation of this
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
#env = JoypadSpace(env, RIGHT_ONLY)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = apply_wrappers(env)


env.reset()
next_state, reward, done, trunc, info = env.step(action=0)


check_points = [722,898,2130,5000]    #Pixel values for n checkpoints. NUM_RUNS ~ NUM_TRIALS**n (not exact depending on checkpoint placment)
check_points = [434, 942, 1596, 1887, 2130, 2594, 5000] 
data_collect = HumanDataCollect(check_points, 79 ,NUM_TRIALS) #Object used to collect data

if DATA_COLLECT:

    for i in range(NUM_TRIALS):    
        print("Trial:", i)
        done = False
        truncated = False
        state, _ = env.reset()
        print("Trial finished, resetting...")
        total_reward = 0
        prev_info = None

        xmax = 0
        while not done and not truncated:
            a = get_action_from_keys()
            state, reward, done, truncated, info = env.step(a)
            #print(info['x_pos'])
           
            data_collect.save_action( a, info, done, truncated)
    
            env.render()
            time.sleep(0.02)
    
    
    data_collect.store_actions()
    
data_collect.load_actions()

########################################################################################
########################################################################################
#########Agent Clones My Playstyle Using VAE
import random
from action_VAE import ActionVAE
import torch.nn.functional as F
import math
#################################### loss function
############# Weighted KL divergence using average reward as a metric

toggle_loss_func = 150     #changes the loss for different era of training

def reward_biased_vae_loss(original_actions, reconstructed_actions, mu, logvar, mean_reward, batch_reward, beta):
    """
    Reward-biased VAE loss.

    Parameters:
    - original_actions: torch.Tensor [batch, seq_len, action_dim]
    - reconstructed_actions: torch.Tensor [batch, seq_len, action_dim]
    - mu, logvar: latent parameters from encoder
    - mean_reward: running average reward (baseline)
    - batch_reward: reward for this batch/segment

    Returns:
    - loss: scalar tensor
    """
    global toggle_loss_func
    
    # Ensure actions are tensors
    if not torch.is_tensor(original_actions):
        original_actions = torch.tensor(original_actions, dtype=torch.float32)
    if not torch.is_tensor(reconstructed_actions):
        reconstructed_actions = torch.tensor(reconstructed_actions, dtype=torch.float32)

    # Reconstruction loss
    #recon_loss = F.mse_loss(reconstructed_actions, original_actions, reduction="mean")
    recon_loss = F.cross_entropy(reconstructed_actions, original_actions)
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original_actions.size(0)


    # Final loss
    if toggle_loss_func>0:
        # Bias term (positive if better than average)
        sigma = max(mean_reward/4, 1.0)  # prevents division by tiny numbers
        reward_diff = torch.tensor(batch_reward - mean_reward)
        reward_bias = torch.tanh(reward_diff/sigma)
        loss = reward_bias * recon_loss + beta*kl_loss
    else:
        toggle_loss_func -= 1
        loss = recon_loss + beta*kl_loss

    return loss

#################################### end loss function

## Train parameters and initialization
SHOULD_CLONE = True 
NUM_OF_EPISODES = 2000      # how many trials

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available(): #device agnostic
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## nn parameters and defintion
action_dim = env.action_space.n
input_dims=env.observation_space.shape
latent_dim = 42*42
sm1_policy = ActionVAE(input_dims, action_dim, latent_dim, hidden_dim1=32, hidden_dim2=32).to(device)
sm1_policy.train()
optimizer = torch.optim.Adam(sm1_policy.parameters(), lr=1e-3)

batch_size = 16                 #This is the time the VAE trains on
time_action = []                #This saves the action for each frame
time_states  = []               #This saves the state for each frame
NUM_OF_TRIALS = 200
time_collected = 0

loss = 0
beta = .5                       #Beta-KL divergence, ramps up beta in: loss = reconstruct+beta*KL_div
step = .55/NUM_OF_EPISODES      #first reconstructs playthrough (Beta small) then collapse to Guassian


avg_reward = 0          #this will be the sum of the total awards
n_avg = 0               #this will divide the above

eps = .75            #likelihood for monte-carlo determination of if the playthrough is "pure" (all same trial)

##Begining of cloning loop
if SHOULD_CLONE:
    for i in range(NUM_OF_EPISODES):
        
        print("Episode:", i)
        done = False
        beta += step
        
        state, _ = env.reset()
        prev_info = None
 
        wtf = 0
        xmax = 0
        break_j = False

        total_reward = 0
        
        mc = random.random() 
        if mc < eps:
            trial_number = [random.randint(0, NUM_TRIALS-1) for _ in range(len(check_points))]
            print('------------------Mixed')
        else:
            trial_number = [random.randint(0,NUM_TRIALS-1)]*len(check_points)
            print('------------------Pure')


##########Epsiode begins and only ends when the level is complete or mario dead
        while not(done):
            

            for j,ll in enumerate(trial_number):
                #print(f'check point: {j} trial: {ll}')
                if(break_j):
                    break_j = False
                    break
                
                actions = data_collect.actions_segments[j][ll]
                
                for a in actions:
                    
                    reward = 0
                    new_state, _ , done, truncated, curr_info = env.step(a)
                    
                    if time_collected == 0:
                        x_start = curr_info['x_pos']
                    
                    
                    time_collected += 1
                    time_action.append(a)
                    time_states.append(state)
    
                    if prev_info is not None:
                      reward = sm1_policy.evaluate_playthrough(state, new_state, a, curr_info, prev_info, done)
                      total_reward += reward
                      
                    if time_collected%batch_size == 0 or done:
                        x_end = curr_info['x_pos']

                        dead = not(curr_info['flag_get']) and done

                        total_reward += sm1_policy.reward_for_trouble_segment(x_start, x_end, dead)
      
                        
                        state_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in time_states]).to(device)                    
                        time_states.clear()
                        pred_action, mu, logvar = sm1_policy.forward(state_tensor)
    
                        
                        time_action_tensor = torch.tensor(time_action, dtype=torch.float32).to(device) #target actions
                        time_action_tensor = torch.tensor(time_action, dtype=torch.long).to(device)
                        #time_action_tensor = F.one_hot(time_action_tensor, num_classes=action_dim).float()
                        time_action.clear()
                     
#########Nueral Net updates to parameters
                        if not(n_avg==0): 
                            loss = reward_biased_vae_loss(time_action_tensor, pred_action, mu, logvar, avg_reward/n_avg, total_reward, beta)
                           # print(f"total_reward - {total_reward} and avg_reward - {avg_reward/n_avg}")
                            print(f"loss - {loss}")
                        else: 
                            loss = reward_biased_vae_loss(time_action_tensor, pred_action, mu, logvar, 0, total_reward, beta)
                           # print(f"total_reward - {total_reward} and avg_reward - {n_avg}")
                            print(f"loss - {loss}")
                            
    
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
        
                        time_collected = 0
                        avg_reward += total_reward
                        n_avg +=1
                        total_reward = 0
##########
    
                    if done:
                        break_j = True
                        break
                    
                    state = new_state
                    prev_info = curr_info

    # -------- Save after cloning loop --------
    sm1_policy.save(filename="VAE_jumps_runs.pth")

# -------- Load if cloning skipped --------
else:
    sm1_policy.load(filename="VAE_jumps_runs.pth",map_location=device)


env.close()
