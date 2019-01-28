import numpy as np
from collections import namedtuple
import random

class Memory:        
    def __init__(self):                        

        self.memory = []
        self.experience = namedtuple('Experience', 
            field_names=['actor_state', 'critic_state', 'action', 'log_prob', 'reward'])

    def add(self, actor_state, critic_state, action, log_prob, reward):
        """Add a new experience to memory."""        
        e = self.experience( actor_state, critic_state, action, log_prob, reward )
        self.memory.append(e)

    def experiences(self, clear=True):
        actor_states = np.vstack([e.actor_state for e in self.memory if e is not None])
        critic_states = np.vstack([e.critic_state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        log_probs = np.vstack([e.log_prob for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
                
        n_exp = len(self)

        if clear:
            self.clear()

        return actor_states, critic_states, actions, log_probs, rewards, n_exp
    
    def delete(self, i):
        del self.memory[i]

    def clear(self):
        self.memory.clear()
    
    def __len__(self):    
        return len(self.memory)
