import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from collections import defaultdict
import math
from collections import deque

class ActionVAE(nn.Module):
    def __init__(self,input_shape, action_dim, latent_dim, hidden_dim1=32, hidden_dim2=32):
        super().__init__()
        
        # reward parameters for jumping for a long time
        self.in_air = False
        self.airborne_timer = 0.0
        self.delta_t = 1             # time between frames
        self.jump_rewarded = False
        
        # reward parameters for moving foward quick
        buffer_size = 3
        self.frame_buffer = deque(maxlen=buffer_size)
        self.jump_peak_y = 0        # All around heighiest of all heigiets
        self.y_flight = 0.0         # height reached in this jump --both measured from take off y
        self.in_air = False
        
        
        self.death_locations = defaultdict(int)
        self.latent_dim = latent_dim
    
        # --- CNN Encoder for 1-channel pixel stack ---
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        cnn_output_dim = self._get_conv_out(input_shape)

    
        # --- Fully connected layers after CNN ---
        self.encoder_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
    
        # --- Decoder (unchanged) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim)
        )

    def encode(self, x):
        # Pass through CNN
        h = self.cnn(x)
        h = h.view(h.size(0), -1)  # flatten
        # Pass through FC layers
        h = self.encoder_fc(h)
        # Compute latent mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


    def _get_conv_out(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))

### rewards   
    def evaluate_playthrough(self, state, new_state, action, curr_info, prev_info, done):
        """
        Evaluates a single playthrough step using Gym Retro's info dict.
        """
    
        # --- Extract info from current and previous step ---
        x_pos_curr = curr_info.get("x", 0)
        x_pos_prev = prev_info.get("x", 0) if prev_info else x_pos_curr
        
        # --- Extract info from current and previous jump ---
        y_pos_curr = curr_info.get("y", 0)
        y_pos_prev = prev_info.get("y", 0) if prev_info else y_pos_curr
        if y_pos_curr >   self.jump_peak_y:
            self.jump_peak_y = y_pos_curr
    
        health_curr = curr_info.get("health", 1)
        health_prev = prev_info.get("health", 1) if prev_info else health_curr
    
        reached_flag = curr_info.get("flag", 0)
    
        # --- Reward logic ---
        reward = 0.0
    
        # Big reward for reaching the flag
        if reached_flag:
            reward += 50.0
    
        # Penalty for dying / episode end
        elif done:
            self.death_locations[x_pos_curr] += 1
            reward -= 25.0
    
        # Penalty for losing health
        elif health_curr < health_prev:
            reward -= 5.0
    
        # Small living reward for surviving
        else:
            reward += 0.5
    
        # Progress reward based on distance moved forward
        distance_progress = x_pos_curr - x_pos_prev                             # reward forward motion and 
        distance_progress = distance_progress if distance_progress>0 else -.25  # penalize moving backwards and staying still
        jump_progress     = max(0, y_pos_curr - y_pos_prev)  # only reward jumping and not falling
        
        
        ## This segment adds up all the rewards due to jumping
        lambda_ = .5
        y_thresh = self.jump_peak_y/10 
        k = 1.5
        reward += .5* distance_progress + 5*jump_progress + lambda_ * (math.exp(k * (self.jump_peak_y - y_thresh)) - 1)  
        reward += self.update( abs(y_pos_curr-y_pos_prev)!=0,y_pos_curr)
    
        return reward
    
    
    def reward_for_trouble_segment(self, x_start, x_end, dead):
        """
        Compute a reward proportional to death counts in a given x-range and decay them.
        Adds a buffer around x_start and x_end to account for Mario's width.
        Can skip reward if Mario died in the segment.
        
        Args:
            death_locations (dict): {x_pos: death_count}
            x_start (float): starting x position of segment
            x_end (float): ending x position of segment
            scale (float): multiplier for reward
            decay (float): fraction to decrease death count after passing
            dead (bool): if True, do not give reward
        
        Returns:
            reward (float): reward for passing the segment
        """
        # Skip reward if Mario died in this segment
        if dead:
            return 0.0
    
        # Hardcoded buffer for Mario's width in pixels
        buffer = 8  # example: 8 pixels; adjust as needed
        x_start_buf = x_start
        x_end_buf = x_end - buffer
    
        # Find keys in range including buffer
        keys_in_range = [k for k in self.death_locations.keys() if x_start_buf <= k <= x_end_buf]
        
        # Compute reward proportional to deaths
        reward = sum(self.death_locations[k] for k in keys_in_range) * .75
        
        # Decrease death counts
        for k in keys_in_range:
            self.death_locations[k] = max(0, self.death_locations[k] - .5)
            
        
        return reward
    
    def update(self, on_ground, y):
          """
          on_ground: boolean from environment info
          y: current vertical position
          returns: reward for this frame (0 if no jump finished)
          """
          reward = 0.0
    
          if not on_ground:
              # In the air
              if not self.in_air:
                  # Jump just started
                  self.in_air = True
                  self.airborne_timer = 0.0
                  self.jump_rewarded = False
                  self.takeoff_y = y
                  self.y_flight = 0.0
    
              # Track the current flight height
              self.y_flight = max(self.y_flight, y - self.takeoff_y)
    
              # Update the *global* jump_peak_y if this jump beats the record
              if self.y_flight > self.jump_peak_y:
                  self.jump_peak_y = self.y_flight
    
              # Increment air time
              self.airborne_timer += self.delta_t
    
          else:
              # On the ground
              if self.in_air and not self.jump_rewarded:
                  # Jump just ended
                  k_airborne = 0.75
                  reward = k_airborne * self.airborne_timer
                  self.jump_rewarded = True
    
              # Reset per-jump variables
              self.in_air = False
              self.airborne_timer = 0.0
    
          return reward

        
 # ---------------- Save / Load Functions ----------------
    def save(self, filename="vae_save.pth", optimizer=None):
        # Save inside ./model_save/
        save_dir = os.path.join(os.getcwd(), "model_save")
        os.makedirs(save_dir, exist_ok=True)
    
        path = os.path.join(save_dir, filename)
        data = {'model_state_dict': self.state_dict()}
    
        # Optional optimizer save (for reproducibility)
        if optimizer is not None:
            data['optimizer_state_dict'] = optimizer.state_dict()
    
        torch.save(data, path)
        print(f"✅ Model saved to {path}")
    
    
    def load(self, filename="vae_save.pth", optimizer=None, map_location=None):
        # Load from ./model_save/
        load_dir = os.path.join(os.getcwd(), "model_save")
        path = os.path.join(load_dir, filename)
    
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data['model_state_dict'])
    
        if optimizer is not None and 'optimizer_state_dict' in data:
            optimizer.load_state_dict(data['optimizer_state_dict'])
    
        print(f"✅ Model loaded from {path}")