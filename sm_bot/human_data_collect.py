import numpy as np
import os

import pickle

from collections import defaultdict

class HumanDataCollect:
    #saves actions
    def __init__(self,
                 check_points,
                 y_check_point,
                 num_episodes):
        
        
        self.num_episodes = num_episodes                 # sets the amount of playthroughs/training data 
        self.check_points = check_points

        self.check_point_counter = 0                     # used to keep track of actions that are saved under
        self.episode_counter = 0                         # a given checkpoint and playthrough
        
        self.y_check_point = y_check_point               # checkpoints above only give x-pos. This is an option  
                                                         # to require mario to pass through a given y-pos too
                                                                
        self.actions_segments = defaultdict(lambda: defaultdict(list))  # how actions are organized for replay to train

                

    def save_action(self, action, info, done, truncated):
        
        if(done or truncated):              #truncate is an extra condition of env.step(), which I did not use in this loop 
            self.episode_counter += 1       #when done with trial we are on next episode and first checkpoint
            self.check_point_counter = 0
            return None
        
        x = info['x_pos']
        y = info['y_pos']
        #print(x)
        
        self.actions_segments[self.check_point_counter ][self.episode_counter].append(action)  #save action segments for future sampeling
        
     
        x_check = self.check_points[self.check_point_counter]    #current checkpoint
        if(x>=x_check):
            if(x_check == x_check and y == self.y_check_point):  #if we pass the current checkpoint, update checkpoint
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
            
        
