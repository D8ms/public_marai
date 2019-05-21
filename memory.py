import random
import numpy as np
from copy import deepcopy
class Memory:
    def __init__(self, num_env, capacity, num_features):
        self.num_env = num_env
        self.states = np.zeros((num_env, capacity, 84, 84, 3), dtype=np.uint8)
        self.tail_states = np.zeros((num_env, 1, 84, 84, 3), dtype=np.uint8)
        self.action_indexes = np.zeros((num_env, capacity), dtype=np.int32)
        self.action_probs = np.zeros((num_env, capacity), dtype=np.float32)
        
        #used to calculate surprisal, which i sused to advantage and qsa value.
        #not directly used in loss function
        self.predicted_rewards = np.zeros((num_env, capacity), dtype=np.float32)

        #these are provided all at once
        self.advs = None
        self.rews = None
        
       
        self.index = 0
        self.capacity = capacity 
        self.num_features = num_features

    def save_states(self, states):
        self.states[:, self.index] = states

    def save_tail_states(self, states):
        self.tail_states[:, 0] = states

    def save_action_indexes(self, action_indexes):
        self.action_indexes[:, self.index] = action_indexes

    def save_action_probs(self, probs):
        self.action_probs[:, self.index] = probs

    def save_predictions(self, predicted_rewards):
        self.predicted_rewards[:, self.index] = predicted_rewards

    def save_advs_and_rews(self, advs, rews):
        #make sure they are numpy arrays
        self.advs = advs
        self.rews = rews

    def inc_index(self):
        self.index = self.index + 1

    def clear(self):
        self.__init__(self.num_env, self.capacity, self.num_features)
   
    def get_by_indexes(self, indexes):
        return [
            np.concatenate(self.states[indexes], axis=0), 
            self.action_indexes[indexes].flatten(),
            self.action_probs[indexes].flatten(),
            np.concatenate(self.tail_states[indexes], axis=0),
            self.advs[indexes].flatten(),
            self.rews[indexes].flatten()
        ]
