import numpy as np
import sys 

class RewardProcessor:
    def __init__(self, num_env, rollout_size):
        self.num_env = num_env
        self.rollout_size = rollout_size
        #Paper uses RewardForwardFunction.rewems
        self.sum_discounted_rewards = None

        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 1e-4 # why cant this be 0?

    def calc_normalized_advantages_and_rewards(self, predicted_rewards, tail_predicted_rewards, surprisals, gamma=0.99, lam=0.95):
        norm_surprisals = self.normalize_surprisals(surprisals, gamma)
        print("norm surprisals", np.mean(norm_surprisals))
        advs, rews = self.calculate_advantages_and_rewards(predicted_rewards, tail_predicted_rewards, norm_surprisals, gamma, lam)
        return self.normalize_advantages(advs), rews
    
    def update_sum_discounted_rewards(self, rewards, gamma):
        if self.sum_discounted_rewards is None:
            self.sum_discounted_rewards = rewards
        else:
            self.sum_discounted_rewards = self.sum_discounted_rewards * gamma + rewards
        return self.sum_discounted_rewards
            

    def normalize_surprisals(self, surprisals, gamma):
        transposed = surprisals.T
        rffs = np.array([self.update_sum_discounted_rewards(rew, gamma) for rew in surprisals.T])
        raveled = rffs.ravel()
        batch_mean = np.mean(raveled)
        batch_variance = np.var(raveled)
        print("mean rew", np.mean(surprisals))
        print("rewems", self.sum_discounted_rewards)
        print("mean, var", batch_mean, batch_variance)
        batch_count = raveled.size
        running_std = self.update_running_std(batch_mean, batch_variance, batch_count)
        return surprisals / running_std

    def update_running_std(self, batch_mean, batch_var, batch_count):
        #from baselines which uses this
        #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        total_count = batch_count + self.running_count    

        delta_mean = batch_mean - self.running_mean
        new_mean = self.running_mean + delta_mean * batch_count / total_count
        
        weighted_running_var = self.running_var * self.running_count
        weighted_batch_var = batch_var * batch_count
        new_var = (weighted_running_var + weighted_batch_var + np.square(delta_mean) * self.running_count * batch_count / total_count) / total_count
        
        self.running_mean = new_mean
        self.running_var = new_var
        self.running_count = total_count
        return np.sqrt(self.running_var)

    def running_std(self):
        return np.sqrt(self.running_var)

    def normalize_advantages(self, advs):
        mean = np.mean(advs)
        std = np.std(advs)
        return (advs - mean) / (std + 1e-7)
    
    #reward is Q(s,a)
    #predicted_rewards V(s)
    #Reminder that  A(s, a) = Q(s, a) - V(s)
    #or             Q(s, a) = A(s, a) + V(s)
    def calculate_advantages_and_rewards(self, predicted_rewards, tail_predicted_rewards, norm_surprisals, gamma, lam):
        delta_marginal_rews = np.zeros((self.num_env, self.rollout_size), dtype=np.float32)
        advs = np.zeros((self.num_env, self.rollout_size), dtype=np.float32)
        next_predicted_rewards = tail_predicted_rewards
        last_adv = 0
        for i in reversed(list(range(norm_surprisals.shape[-1]))):
            delta_marginal_rews[:, i] = norm_surprisals[:, i] + (gamma * next_predicted_rewards) - predicted_rewards[:, i]
            next_predicted_rewards = predicted_rewards[:, i]
            advs[:, i] = last_adv = delta_marginal_rews[:, i] + gamma * lam * last_adv
        rews = np.sum([advs, predicted_rewards], axis=0)
        return advs, rews
