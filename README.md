This is my first uploaded project to github. It is a work in progress. I want to add a nueral network of somekind (probably VAE) to provide a latent representation of my play style, and also a custom policy learned from my gameplay. These parts are in the works. The main code should also have enough comments to follow along, but an outline is given here.

##### DESCRIPTION OF IMPLEMENTATION OF AI AGENT

The agent is implemented with a double-Q learning algorithm using the Bell-man equation. A convolutional nueral network is trained on the game state and the resultant reward based off the policy where motion is encoded in the training by stacking 4 consecutive frames of gameplay in each training step. An epsilon-greedy approach is used to explore the action space where the bot performs a random action if a rand_int<epsilon is choosen; otherwise, the optimal action determined by the cnn is taken. Taking the optimal action is a greedy apporach, so to insetivize reward a replay buffer of the above cnn is saved every num_save epochs of training and is used in the above Bell-man equation to determine the loss of our current nework. Notice that this means the ccn after training picks an optimal step as a greedy algorithm would, but this choice is crafted around future reward. 

The output space is simply the number of button combinations the bot is allowed to hit (taken to be a subset of a human players) and the frames are pixelated to a lower resolution as well as greyscaled to reduce the number of color channels in the ccn. To train the ai agent the main program needs to be ran. It forms an instance of smw_gym_retro and the agent of the Agent class using the double-Q learning algorithm. There are several modules present before this exploratory loop. 

First data is collected by capturing real time gameplay using a class HumanDataCollect to train a variational autoencoder (VAE) to learn the sematics of the expert users play style. The class saves a bunch of segments set by checkpoints for a set of playthroughs and combines them to form a statistical ensemble. The main purpose is to train mario to move foward and run. A convolutional nueral network will be used to acquire the inputs that are feed into the VAE. The button combinations allowed by the bot from gym_retro include simple, normal, and complex. My goal is to use complex button combinations and reduce the training time for the double-Q learning algorithm to acquire the knowledge mario should move foward and also he should jump high while moving.  
**Note a double Q-learning algorithm can also be trained by using these segments, just drop the epsilon greedy approach till the exploration stage. 


#### IMPORTANT MODEL PARAMETER DESCRIPTION
#### How to use the parameters of the maincode. Code snippets will be distinguished from text with a # to the left of it

#### DATA COLLECTION FROM MY PLAYSTYLE
To record playthroughs or not. This module will take inputs from a Human player in order to extract the latent information in the experts playstyle
#### *DATA_COLLECT = True/False*

These will set total number of playthroughs or trials and the location of checkpoints. In total there will be NUM_TRIALS**(length(checkpoints)) playthroughs to sample from in the training group if we randomly select a trail for a given segment.
Notice that the locations of the checkpoints could introduce a bias into the training. The hopes is the VAE below abstracts these issues away.
#NUM_OF_TRIALS = 7
#check_points = [722,898,2130,5000]

This class saves the playthrough actions to a dictionary to sample from for the training loop
#data_collect = HumanDataCollect(check_points, 79 ,NUM_OF_TRIALS)

#### AGENT CLONES MY PLAY STYLE

This will load data from the previous
#data_collect.load_actions()

Should clone my playstyle or not
#SHOULD_CLONE = False

This will set the number of episodes to simulate
#NUM_OF_EPISODES = 2000

#ENV_NAME = 'SuperMarioBros-1-1-v0'


#DISPLAY = True
#env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
