import numpy as np

import torch

from memory import Memory

class Agent:

    def __init__(
        self, 
        device,
        key,
        actor_model,
        n_step,       
        ):

        self.DEVICE = device
        self.KEY = key

        # NEURAL MODEL
        self.actor_model = actor_model   

        # MEMORY
        self.memory = Memory()

        # HYPERPARAMETERS
        self.N_STEP = n_step
        

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():                
            action, log_prob, _ = self.actor_model(state)                    
        self.actor_model.train()

        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()

        return action, log_prob

    def step(self, actor_state, critic_state, action, log_prob, reward):                
        self.memory.add( actor_state, critic_state, action, log_prob, reward )