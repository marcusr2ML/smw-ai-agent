This is my first uploaded project to github. It is a work in progress. I want to add a nueral network of somekind (probably VAE) to provide a latent representation of my play style, and also a custom policy learned from my gameplay. These parts are in the works


The agent is implemented with a double-Q learning algorithm using the Bell-man equation. A convolutional nueral network is trained on the game state and the resultant reward based off the policy. A stack of 4 frames are also used on each step of training to simulate motion and a replay buffer the action of the bot is taken as the most likely input. The input space is simply the number of button combinations the bot is allowed to hit (being a subset of a human player).  with a epsilon-decay
