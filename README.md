# **Super Mario AI Agent Project**

This is my first uploaded project to GitHub. It is a work in progress. I want to add a neural network (probably **VAE**) to provide a latent representation of my play style, and also a custom policy learned from my gameplay. These parts are in the works. The main programs have enough comments to follow along, and an outline is given below.

#### **Main Programs (as of Nov. 2, 2025)**
* **main_sm1_VAE.py** — clones my playstyle using replay buffers and a **VAE**
* **main_sm1_DQ.py** — clones my playstyle using replay buffers and **Double-Q learning** on my actions followed by an **epsilon-greedy algorithm** to explore further

##### **Description of Implementation of AI Agent**
The agent is implemented with a **Double-Q learning algorithm** using the Bellman equation. A **convolutional neural network (CNN)** is trained on the game state and the resultant reward based on the policy, where motion is encoded by stacking 4 consecutive frames of gameplay in each training step.  

An **epsilon-greedy approach** is used to explore the action space: the bot performs a random action if a `rand_int < epsilon`; otherwise, the optimal action determined by the CNN is taken. Taking the optimal action is a greedy approach, so to **incentivize reward**, a replay buffer of the CNN is saved every `num_save` epochs and used in the Bellman equation to determine the loss of the current network. After training, the CNN picks optimal steps as a greedy algorithm would, but choices are crafted to maximize **future reward**.  

The **output space** is the number of allowed button combinations (a subset of human player inputs). Frames are **pixelated and greyscaled** to reduce the number of color channels for the CNN.  

To train the AI agent, the main program must be run. It instantiates **smw_gym_retro** and the **Agent** class using the **Double-Q learning algorithm**.  

Gameplay data is first collected using the **HumanDataCollect** class to train a **VAE** to learn the semantics of expert play style. This class saves multiple segments defined by checkpoints across playthroughs and combines them into a statistical ensemble. The main purpose is to train Mario to **move forward and run**.  

The CNN acquires inputs that are fed into the VAE. Button combinations allowed by the bot from **gym_retro** include simple, normal, and complex. My goal is to use **complex button combinations** and reduce the training time for the Double-Q learning algorithm, so Mario learns to move forward and jump high efficiently.  

**Note:** A **Double-Q learning algorithm** can also be trained directly using these segments — just drop the epsilon-greedy approach until the exploration stage.

#### **Important Model Parameters and Modules**
#### **How to use the parameters of the main code**
Code snippets are distinguished from text with a bullet point.

#### **Data Collection from My Playstyle** (same across main programs)
To record playthroughs or not, set:

* **DATA_COLLECT = True/False**

Set the total number of playthroughs or trials and the locations of checkpoints. In total, there will be `NUM_TRIALS^(length(check_points))` playthroughs sampled randomly during training.  

* **NUM_OF_TRIALS = 7**
* **check_points = [722, 898, 2130, 5000]**

This class saves playthrough actions to a dictionary for training:

* **data_collect = HumanDataCollect(check_points, 79, NUM_OF_TRIALS)**

#### **Agent Clones My Playstyle** (main programs differ slightly)
Load data from the previous module:

* **data_collect.load_actions()**

Should clone my playstyle or not:

* **SHOULD_CLONE = False**

Set the number of episodes to simulate:

* **NUM_OF_EPISODES = 2000**

Set world, stage, and version:  

* **ENV_NAME = 'SuperMarioBros-1-1-v0'**

Training environment instantiation and render mode:

* **DISPLAY = True**
* **env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)**
