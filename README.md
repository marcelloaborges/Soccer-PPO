<img src="https://camo.githubusercontent.com/1471a35ea88bdb4fc0915e31a637cc4295e0771e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4175677573742f35623831636430355f736f636365722f736f636365722e706e67">


# DRL - PPO Algorithm - Soccer Twos
Udacity Deep Reinforcement Learning Nanodegree Program


### Observations:
- To run the project just execute the <b>main.py</b> file.
- If you are not using a windows environment, you will need to download the corresponding <b>"Reacher"</b> version for you OS system. Mail me if you need more details about the environment <b>.exe</b> file.
- The <b>checkpoint.pth</b> has the expected average score already hit.


### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0


## The problem:
- The task envolves a soccer game with 2 teams, each one having 2 players: 1 striker and one 1 keeper.
- There is no goal defined for default, so I decided to train againt a random team until my agents archive a score
of 95 win into 100 games.
- The goalies have 4 actions.
- The strikers have 6 actions.

## The solution:
- The biggest problem in this scenario is to control the exploration vs. explotation rate. I tried approaches
such as Double DQN with an exponential exploration rating decay as well as the DDPG aproach with prioritized replay experience
for diversification of the experiences on learning, but I couldn't find the rigth configuration for the hyperparameters
that could make the agents converge.
- So I changed the approach for a PPO strategy since this kind of method is easier to configure and controls the
exploration very well by itself using probabilistic decisions. After a lot of different implementations I've reached
the current solution.
- There are still some items to improve such as convergence time and multi teams training, but I'm satisfied with
the current results. On my last test I could archive the goal (95 wins in 100 games) with a little more than 5000 episodes
and I consider that a great result if I look back to all the tries I made before.
- It was really good for my learning as I had not used PPO approaches at this level before trying this environment and
for sure the knowledge acquired here will be very relevant for my next projects.
- Talking about the implementation, it has an actor critic neural model and is using a proximal policy optimization
learning function with the trusted region approach. The learning happens after each episode (controlled by the environment),
and it uses mini-batches from the episode experiences after the reward calculation using the N-Step method that combines
the temporal difference discount with monte carlo tree search exploration (in this case the N-Step range is the role episode).
- For now, I'll try other variations changing when the learning happens and using multi teams for experience gathering.
I hope I can archive superhuman results with 5000 episodes or less (the agents are good but not super humans with 5000 episodes).
- One last consideration. To beat a random team looks easier at the beginning, but if you consider that random agents wins 1/3
of the games and the draw rate of random games is 1/3, the AI has overcome a big challenge reaching 95% win rate. It's incredible
how a random agent can score with just a few steps.


### The hyperparameters:
- The file with the hyperparameters configuration is the <b>main.py</b>. 
- If you want you can change the model configuration to into the <b>model.py</b> file.
- The actual configuration of the hyperparameters is: 
  - Learning Rate Goalie: 8e-5
  - Learning Rate Striker: 1e-4
  - Gamma: 0.995
  - Batch Size: 32
  - Epsilon: 0.1
  - Entropy Weight: 0.001

- For the neural models:    
  - Actor    
    - Hidden: (input, 256)          - ReLU
    - Hidden: (256, 128)            - ReLU
    - Output: (128, action_size)    - Softmax

  - Critic
    - Hidden: (input, 256)          - ReLU
    - Hidden: (256, 128)            - ReLU
    - Output: (128, 1)              - Linear
