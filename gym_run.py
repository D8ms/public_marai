from drivers import GymDriver
from client import NesClient
from model import Model
import numpy as np
import cv2
from memory import Memory
import tensorflow as tf
from PIL import Image
from reward_processor import RewardProcessor
import os

epoch_no = 0
num_stacked_frames = 3
num_skipped_frames = 5
rollout_size = 128
mini_batch_size = 1
epoch_per_rollout = 3
num_env = 8
all_visited_levels = set()

driver = GymDriver(num_env, 10000)
memory = Memory(num_env, rollout_size, 512)
rp = RewardProcessor(num_env, rollout_size)
model = Model(driver.num_actions, driver.frame_mean, driver.frame_std)

train_counter = 0
tf.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
tf.config.operation_timeout_in_ms = 60000

with tf.Session(config=tf.config) as sess:
    sess.run(tf.global_variables_initializer())
    summary_recorder = tf.summary.FileWriter("summary", sess.graph)
    while True:
        f = 0
        episode_infos = []
        while f < rollout_size:
            frames, selected_move_indexes, action_probabilities, predicted_rewards, infos = driver.observe_act(model, sess)
            memory.save_states(frames) 
            memory.save_action_indexes(selected_move_indexes)
            memory.save_predictions(predicted_rewards.flatten())
            memory.save_action_probs(action_probabilities.flatten())
            memory.inc_index()
            
            if infos is not None:
                episode_infos.append(infos)
            f += 1
        
        tail_states = driver.observe()
        memory.save_tail_states(tail_states)
        _, _, tail_predicted_rewards = model.forward(sess, tail_states)
        tail_predicted_rewards = np.array([x[0] for x in tail_predicted_rewards])
        surprisals = np.zeros((num_env, rollout_size), dtype=np.float32) 
        for i in range(num_env):
            surprisals[i] = model.get_surprisal(sess, memory.states[i], memory.action_indexes[i], memory.tail_states[i])
        normalized_advantages, rewards = rp.calc_normalized_advantages_and_rewards(memory.predicted_rewards, tail_predicted_rewards, surprisals)
        memory.save_advs_and_rews(normalized_advantages, rewards)
        print("reward mean std", np.mean(rewards), np.std(rewards))
        for ei in episode_infos:
           for info in ei:
                visited_levels = info['levels'] if 'levels' in info else set()
                all_visited_levels = all_visited_levels.union(visited_levels)
        print("visited_levels")
        print(all_visited_levels)
        
        for _ in range(epoch_per_rollout):
            random_indexes = np.arange(num_env)
            np.random.shuffle(random_indexes)
            for start in range(0, num_env, mini_batch_size):
                end = start + mini_batch_size #python wont complain if end is above the max index of array[start:end]
                _states, _action_indexes, _action_probabilities, _tail_states, _advantages,  _predicted_rewards = memory.get_by_indexes(random_indexes[start:end])
                summary = model.train(sess, _states, _action_indexes, _action_probabilities, _tail_states,  _advantages,  _predicted_rewards)
                summary_recorder.add_summary(summary, epoch_no)
                epoch_no += 1
        print(epoch_no, " finished")
        memory.clear()
