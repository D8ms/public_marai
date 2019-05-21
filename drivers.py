import os
from lsc_utils.lsc_vec_env import ShmemVecEnv as VecEnv
import lsc_utils.lsc_wrappers as lsc_wrappers
import numpy as np
from controller import Controller
from client import NesClient
import cv2
import sys

class GymDriver:
    def __init__(self, num_envs, num_rand_moves):
        self.cached_obs = None
        self.first_obs = True

        dummy_env = lsc_wrappers.make_mario_env()

        self.ob_space = dummy_env.observation_space
        self.ac_space = dummy_env.action_space
        self.init_mean_std(dummy_env, num_rand_moves)
        #print("spaces", ob_space, ac_space)
        self.num_actions = self.ac_space.n
        del dummy_env
        self.init_envs(num_envs, self.ob_space, self.ac_space)
        
    def init_mean_std(self, dummy_env, num_rand_moves):
        obs = [np.asarray(dummy_env.reset())]
        for _ in range(num_rand_moves):
            ac = dummy_env.action_space.sample()
            ob, _, done, _ = dummy_env.step(ac)
            if done:
                ob = dummy_env.reset()
            obs.append(np.asarray(ob))

        self.frame_mean = np.mean(obs, 0).astype(np.float32)
        self.frame_std = np.std(obs, 0).mean().astype(np.float32)

    def init_envs(self, num_env, ob_space, ac_space):
        env_fns = [lsc_wrappers.make_mario_env for i in range(num_env)]
        self.envs = VecEnv(env_fns, spaces=[ob_space, ac_space])

    def observe_act(self, model, sess):
        if self.first_obs:
            frames = self.envs.reset()
            infos = None
            self.first_obs = False
        elif self.cached_obs is None:
            frames, prevrews, news, infos = self.envs.step_wait()
        else:
            frames = self.cached_obs
            infos = None
            self.cached_obs = None
            
        selected_move_indexes, action_neglogps, predicted_rewards = model.forward(sess, frames) 

        self.envs.step_async(selected_move_indexes)
        return frames, selected_move_indexes, action_neglogps, predicted_rewards, infos

    def observe(self):
        self.cached_obs = self.envs.step_wait()[0]
        return self.cached_obs

class Driver:
    def __init__(self, num_envs, starting_level, save_state_path, num_rand_moves, num_stacked_frames, num_skipped_frames, use_cache, record):
        self.cached_obs = None
        self.first_obs = True
        self.num_stacked_frames = num_stacked_frames
        self.num_skipped_frames = num_skipped_frames
        
        self.agent = Controller() 
        dummy_env = NesClient(starting_level, save_state_path).init_mario()
        if os.path.exists("mario_mean.npy") and os.path.exists("mario_std.npy") and use_cache:
            self.frame_mean = np.load("mario_mean.npy")
            self.frame_std = np.load("mario_std.npy")
        else:
            print("Generating mean and std, this may take a minute, please wait warmly.")
            self.init_mean_std(dummy_env, num_rand_moves)
        del dummy_env

        #print("spaces", ob_space, ac_space)
        self.num_actions = len(self.agent.all_moves())
        self.init_envs(num_envs, starting_level, save_state_path)

        if record:
            for i in range(len(self.envs)):
                print("replays online")
                env = self.envs[i]
                env.enable_recorder(i)
        
    def init_mean_std(self, dummy_env, num_rand_moves):
        obs = []
        moves = self.agent.rand_move(num_rand_moves)

        for move in moves:
            ob = self.observe([dummy_env], cache=False)[0]
            obs.append(np.asarray(ob))
            action_bin = self.agent.all_moves()[move]
            dummy_env.set_action(action_bin)
        self.frame_mean = np.mean(obs, 0).astype(np.float32)
        self.frame_std = np.std(obs, 0).mean().astype(np.float32)
        np.save("mario_mean", self.frame_mean)
        np.save("mario_std", self.frame_std)

    def init_envs(self, num_env, starting_level, save_state_path):
        self.envs = []
        for i in range(num_env):
            self.envs.append(NesClient(starting_level, save_state_path).init_mario())

    def observe_act(self, model, sess, envs=None):
        if envs is None:
            envs = self.envs
        if self.cached_obs is None:
            frames = self.observe(envs, cache=False)
        else:
            frames = self.cached_obs
            self.cached_obs = None
            
        selected_move_indexes, action_neglogps, predicted_rewards = model.forward(sess, frames)
        for i in range(len(envs)):
            move_idx = selected_move_indexes[i]
            action_bin = self.agent.all_moves()[move_idx]
            envs[i].set_action(action_bin)
        env_visited_levels = [env.visited_levels for env in envs]
        all_visited_levels = set.union(*env_visited_levels)
        return frames, selected_move_indexes, action_neglogps, predicted_rewards, all_visited_levels

    def observe(self, envs=None, cache=True):
        if envs is None:
            envs = self.envs
        [env.prepare_advance_state() for env in envs]
        for i in range(self.num_stacked_frames):
            for j in range(self.num_skipped_frames):
                [env.step_skipped_frame() for env in envs]
            [env.step_observed_frame(preprocess_frame_func) for env in envs]
        stacked_frames = [env.get_observation() for env in envs]
        if cache:
            self.cached_obs = stacked_frames
        [env.update_pb_info() for env in envs]
        return stacked_frames

def preprocess_frame_func(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(grayscale, (84, 110), interpolation=cv2.INTER_AREA)
    return frame[18:102, :]
