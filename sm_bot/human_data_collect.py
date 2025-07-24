import torch
import numpy as np
import os

import pickle

from collections import defaultdict

class HumanDataCollect:
    def __init__(self,
                 check_points,
                 y_check_point,
                 num_episodes,
                 batch_size=32, 
                 sync_network_rate=10000):
        
        
        self.num_episodes = num_episodes
        self.check_point_counter = 0
        self.episode_counter = 0
        self.check_points = check_points
        self.y_check_point = y_check_point
        # Hyperparameters
        self.batch_size = batch_size
        
        self.actions_segments = defaultdict(lambda: defaultdict(list))

                

    def save_action(self, action, info, done, truncated):
        
        if(done or truncated):
            self.episode_counter += 1
            self.check_point_counter = 0
            return None
        
        x = info['x_pos']
        y = info['y_pos']
        #print(x)
        
        self.actions_segments[self.check_point_counter ][self.episode_counter].append(action)
        
        # if(self.check_point_counter<=len(self.check_points)-1 and x>= self.check_points[self.check_point_counter]):
        #     self.check_point_counter +=1
        #     print('check point reached: ', x)
                
        
        d = 4
        x_check = self.check_points[self.check_point_counter]
        if(x>=x_check):
            if(x_check == x_check and y == self.y_check_point):
                self.check_point_counter +=1
                print('check point reached: ', x_check)
            else:
                print('Turn around to set the check point!!' ,\
                      x,',',y, ' --- ' ,self.check_points[self.check_point_counter],\
                     ',', self.y_check_point )
        
        
        return None

    def store_actions(self):
        filename="actions_segments_"+str(len(self.check_points))+"_TRIALS_"+str(self.num_episodes)+".pkl"
        with open(filename, "wb") as f:
            pickle.dump(dict(self.actions_segments), f)
            
            
    def load_actions(self):
        filename="actions_segments_"+str(len(self.check_points))+"_TRIALS_"+str(self.num_episodes)+".pkl"
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
            self.actions_segments = defaultdict(lambda: defaultdict(list), loaded)
            
        