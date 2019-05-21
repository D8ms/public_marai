import numpy as np
import tensorflow as tf
import os
import sys
import argparse

from controller import Controller
from client import NesClient
from model import Model
from reward_processor import RewardProcessor
from memory import Memory
from drivers import Driver

parser = argparse.ArgumentParser(description="Someone sent an Atari to Mars")
parser.add_argument('--level', type=str, default="1-1", help="level to start playing, defaults to '1-1'")
parser.add_argument('--use_cache', default=False, action="store_true", help="Use cached mean and std")
parser.add_argument('--record', default=False, action="store_true", help="Records replay file, states and metadata for video")
opts = parser.parse_args()
save_state_path = "headless_levels/level-" + opts.level + ".state"

if not os.path.exists(save_state_path):
    raise Exception("Could not find save file: ", save_state_path)
print("The starting level is set to: ", save_state_path)

controller = Controller()
epoch_no = 0
num_stacked_frames = 3
num_skipped_frames = 4
rollout_size = 128
mini_batch_size = 1
epoch_per_rollout = 3
num_env = 8
memory = Memory(num_env, rollout_size, 512)
driver = Driver(num_env, opts.level, save_state_path, 10000, num_stacked_frames, num_skipped_frames, opts.use_cache, opts.record)
rp = RewardProcessor(num_env, rollout_size)
all_visited_levels = set()

frame_mean = driver.frame_mean
frame_std = driver.frame_std

model = Model(controller.all_moves().size, frame_mean, frame_std)

train_counter = 0

tf.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
tf.config.operation_timeout_in_ms = 60000

with tf.Session(config=tf.config) as sess:
    #saver = tf.train.Saver(max_to_keep=25)
    sess.run(tf.global_variables_initializer())
    summary_recorder = tf.summary.FileWriter("summary", sess.graph)
    prev_rollout_lastframes = [None] * num_env
    while True:
        f = 0
        while f < rollout_size:
            frames, selected_move_indexes, action_probabilities, predicted_rewards, infos = driver.observe_act(model, sess)
            all_visited_levels = all_visited_levels.union(infos)
            action_bins = controller.all_moves()[selected_move_indexes]
            memory.save_states(frames) 
            memory.save_action_indexes(selected_move_indexes)
            memory.save_predictions(predicted_rewards.flatten())
            memory.save_action_probs(action_probabilities.flatten())
            memory.inc_index()
            f += 1
            
            #save visualization metadata
            for x in range(num_env):
                env = driver.envs[x]
                if env.should_record():
                    env.save_neglogp_metadata(action_probabilities[x])
            #end visualization stuff

        tail_states = driver.observe()
        memory.save_tail_states(tail_states)
        _, _, tail_predicted_rewards = model.forward(sess, tail_states)
        tail_predicted_rewards = np.array([x[0] for x in tail_predicted_rewards])
        surprisals = np.zeros((num_env, rollout_size), dtype=np.float32)
        for i in range(num_env):
            surprisals[i] = model.get_surprisal(sess, memory.states[i], memory.action_indexes[i], memory.tail_states[i])
        normalized_advantages, rewards = rp.calc_normalized_advantages_and_rewards(memory.predicted_rewards, tail_predicted_rewards, surprisals)
        memory.save_advs_and_rews(normalized_advantages, rewards)
        print("advantage mean std", np.mean(normalized_advantages), np.std(normalized_advantages))
        print("reward mean std", np.mean(rewards), np.std(rewards))

        #save visualization metadata
        for x in range(num_env):
            env = driver.envs[x]
            if env.should_record():
                ds_di = model.get_dsurprisal_dinps(sess, memory.states[x], memory.action_indexes[i], [prev_rollout_lastframes[x]])
                da_di = model.get_daction_dinps(sess, memory.states[x], memory.action_indexes[x])
                env.save_surprisals_and_grads_metadata(surprisals[x], ds_di, da_di)
        prev_rollout_lastframes = frames
        #end visualization stuff

        for _ in range(epoch_per_rollout):
            random_indexes = np.arange(num_env)
            np.random.shuffle(random_indexes)
            for start in range(0, num_env, mini_batch_size):
                end = start + mini_batch_size #python wont complain if end is above the max index of array[start:end]
                _states, _action_indexes, _action_probabilities, _next_states, _advantages,  _rewards = memory.get_by_indexes(random_indexes[start:end])
                summary = model.train(sess, _states, _action_indexes, _action_probabilities, _next_states,  _advantages,  _rewards)
                summary_recorder.add_summary(summary, epoch_no)
                epoch_no += 1
        print(epoch_no, " finished") 
        print("visited levels:", all_visited_levels)
        memory.clear()
        sys.stdout.flush()
