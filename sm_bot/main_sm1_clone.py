from pynput import keyboard
import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

from agent import Agent

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
####Data Collection From my playing

DATA_COLLECT = False
ENV_NAME = 'SuperMarioBros-1-1-v0'

NUM_OF_TRIALS = 7
DISPLAY = True
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)

#env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)

#env = JoypadSpace(env, RIGHT_ONLY)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = apply_wrappers(env)


env.reset()
next_state, reward, done, trunc, info = env.step(action=0)


check_points = [722,898,2130,5000]
data_collect = HumanDataCollect(check_points, 79 ,NUM_OF_TRIALS)

if DATA_COLLECT:

    for i in range(NUM_OF_TRIALS):    
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
            # print(info['x_pos'])
           
            data_collect.save_action( a, info, done, truncated)
    
            env.render()
            time.sleep(0.02)
    
    
    data_collect.store_actions()

########################################################################################
########################################################################################
#########Agent Clones My Playstyle
import random 

data_collect.load_actions()

SHOULD_CLONE = False

CKPT_SAVE_INTERVAL = 250
NUM_OF_EPISODES = 2000



model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")
    

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)


if SHOULD_CLONE:
    for i in range(NUM_OF_EPISODES):    
    
        print("Episode:", i)
        done = False
        
        state, _ = env.reset()
        prev_info = None
        
        xmax = 0
        break_j = False
        
        total_reward = 0
        for j in range(len(check_points)):
            
            if(break_j):
                break_j = False
                break
            
            ll = random.randint(0,NUM_OF_TRIALS-1)
            print(ll)
            # ll = i
            actions = data_collect.actions_segments[j][ll]
            for a in actions:
                
                reward = 0
                new_state, _ , done, truncated, curr_info = env.step(a)
                
                #custom reward function
                if prev_info is not None:
                     reward = agent.get_custom_reward(prev_info, curr_info, a, done)
                total_reward += reward
                if done:
                    agent.timer_move = 0       #these give mario insentive to move by tracking consectutive frames where mario is moving or stationary
                    agent.timer_stag = 0
                    agent.x_max = max(agent.x_max,info['x_pos']/2)
                    agent.x_max_curr = 0
                    break_j = True
                    break
        
             
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()
        
                state = new_state
                prev_info = curr_info
        
        if (i + 1) % CKPT_SAVE_INTERVAL == 0:
            print("Saving model:")
            agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))
    
        print("total_reward:", total_reward)
###############################################################################################
###############################################################################################
##########Epsilon Greedy Training

SHOULD_TRAIN = True
NUM_TEST_EPISODES = 5
CKPT_SAVE_INTERVAL_TRAIN = 15


if  SHOULD_TRAIN:
    folder_name = "2025-05-31-19_20_45"         # <-- update with real folder name
    ckpt_name = "model_1500_iter.pt"            # <-- update with actual checkpoint name
    
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 3.0
    agent.eps_min = 05.0
    agent.eps_decay = 999999.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    for i in range(NUM_TEST_EPISODES):    
        print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        prev_info = None

        
        xmax = 0
        while not done:
            a = agent.choose_action(state)
            new_state, total_reward, done, truncated, curr_info = env.step(a)

            # if prev_info is not None:
            #      reward += agent.get_custom_reward(prev_info, curr_info, a)
            # total_reward += reward

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, total_reward, new_state, done)
                agent.learn()

            state = new_state
            prev_info = curr_info
            if done:
                agent.timer_move = 0
                agent.timer_stag = 0
                agent.x_max = max(agent.x_max,info['x_pos']/2)
                agent.x_max_curr = 0
            

        print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL_TRAIN == 0:
            agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

        print("Total reward:", total_reward)

env.close()
