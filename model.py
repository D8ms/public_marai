import tensorflow as tf
from keras import optimizers
import numpy as np
from tensorflow.contrib.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense as KerasDense, Activation, Reshape, Flatten

IN_HEIGHT = 84
IN_WIDTH = 84
IN_DEPTH = 3

def baseline_normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def Dense(units, activation=None, name=None):
    return KerasDense(units, activation, kernel_initializer=baseline_normc_initializer(), name=name)

class SmallConvNet:
    def __init__(self, num_features):
        self.layer_1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu, name="SmallConvNet_conv2d_1")
        self.layer_2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu, name="SmallConvNet_conv2d_2")
        self.layer_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu, name="SmallConvNet_conv2d_3")
        self.layer_4 = Flatten(name="SmallConvNet_flatten")
        self.layer_5 = Dense(num_features, activation=None, name="SmallConvNet_dense")
        
    def attach(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        out = self.layer_5(x)
        return out

class Model:
    def __init__(self, num_actions, frame_mean, frame_std, num_features=512):
        self.num_actions = num_actions
        self.frame_mean = frame_mean
        self.frame_std = frame_std
        self.num_features = num_features
        
        #shared placeholders
        self.frame_ph = tf.placeholder(np.float32, shape=[None, IN_HEIGHT, IN_WIDTH, IN_DEPTH], name="frame_ph")
        self.tail_frame_ph = tf.placeholder(np.float32, shape=[None, IN_HEIGHT, IN_WIDTH, IN_DEPTH], name="tail_frame_ph")
        self.action_taken_ph = tf.placeholder(np.int32, shape=[None], name="action_taken_ph")
        self.reward_ph = tf.placeholder(np.float32, shape=[None], name="reward_ph")
        self.advantage_ph = tf.placeholder(np.float32, shape=[None], name="advantage_ph")
        self.old_action_probability_ph = tf.placeholder(np.float32, shape=[None], name="old_action_probability_ph")
     
        #useful top layer tensors
        self.action_one_hot = tf.one_hot(self.action_taken_ph, self.num_actions)
        self.norm_frame = (tf.to_float(self.frame_ph) - self.frame_mean) / self.frame_std
        self.norm_tail_frame = (tf.to_float(self.tail_frame_ph) - self.frame_mean) / self.frame_std
       
        with tf.variable_scope("dynamics"):
            with tf.variable_scope("shared_features"):
                shared_feature_layer = SmallConvNet(self.num_features)
                self.dynamics_features = shared_feature_layer.attach(self.norm_frame)
                self.tail_dynamics_features = shared_feature_layer.attach(self.norm_tail_frame)
                
                #This will NOT work if we send multiple rollouts per batch
                #Would have to reshape dynamics_feature before concatenating
                #Unless you know a better way to interleave inputs
                self.next_dynamics_features = tf.concat([self.dynamics_features, self.tail_dynamics_features], 0)[1:]

                self.visualize_features = tf.reshape(self.dynamics_features, (-1, 16, 32))
                self.visualize_next_features = tf.reshape(self.next_dynamics_features, (-1, 16, 32))
                self.visualize_tail_features = tf.reshape(self.tail_dynamics_features, (-1, 16, 32))

                #self.next_dynamics_features = tf.concat([self.dynamics_features[1:], self.tail_dynamics_features], 1)
        
            self.forward_dynamic_error, self.forward_dynamic_loss = self.attach_forward_dynamic_head(self.dynamics_features, self.next_dynamics_features)
            self.inverse_dynamic_loss = self.attach_inverse_dynamic_head(self.dynamics_features, self.next_dynamics_features)
        
        with tf.variable_scope("policy"):
            self.policy_convnet = SmallConvNet(self.num_features).attach(self.norm_frame)
            self.action_scores, self.predicted_reward, self.pg_loss, self.reward_loss, self.entropy_loss = self.attach_action_reward_head(self.policy_convnet)
            self.ppo_loss = self.pg_loss + self.reward_loss + self.entropy_loss
        
        self.total_loss = self.ppo_loss + self.inverse_dynamic_loss + self.forward_dynamic_loss
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.total_loss)
    
        self.sample_move_index, self.sample_action_probability = self.attach_random_move(self.action_scores)
    
        self.merged_summary = self.build_summary()

    def attach_inverse_dynamic_head(self, dynamics_features_tensor, next_dynamics_features_tensor):
        #(frame_ph, tail_frame_ph, action_taken_ph) -> inverse_dynamic_loss
        concat_features = tf.concat([dynamics_features_tensor, next_dynamics_features_tensor], axis=-1)
        x = Dense(512, activation=tf.nn.relu)(concat_features)
        my_logits = Dense(self.num_actions, activation=None)(x)
        inverse_dynamic_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=my_logits, labels=self.action_taken_ph))

        return inverse_dynamic_loss

    def attach_forward_dynamic_head(self, dynamics_features_tensor, next_dynamics_features_tensor):
        #(frame_ph, tail_frame_ph, action_taken_ph) -> forward_dynamic_loss
        #This is the "surprisal" reward
        frame_features = tf.stop_gradient(dynamics_features_tensor)
        next_frame_features = tf.stop_gradient(next_dynamics_features_tensor)

        def dense_resnet_layer(tensor):
            #a = Dense(512, activation=tf.nn.leaky_relu)(tf.concat([tensor, self.action_one_hot], axis=-1))
            return Dense(512, activation=None)(tf.concat([tensor, self.action_one_hot], axis=-1))
            #return a + b
        
        with tf.variable_scope("resnet"):
            x = Dense(512, activation=tf.nn.leaky_relu)(tf.concat([frame_features, self.action_one_hot], axis=-1))
            x = dense_resnet_layer(x)
            x = dense_resnet_layer(x)
            x = dense_resnet_layer(x)
            x = dense_resnet_layer(x)
        predicted_next_features = Dense(self.num_features, activation=None)(tf.concat([x, self.action_one_hot], axis=-1))
        
        forward_dynamic_error = tf.reduce_mean(tf.square(predicted_next_features - next_frame_features), -1)
        forward_dynamic_loss = tf.reduce_mean(forward_dynamic_error)


        #these nodes are for saliency
        #we pretend the stop gradient does not next for the next_features to see where we mispredicted
        _fake_fd_error = tf.reduce_mean(tf.square(predicted_next_features - next_dynamics_features_tensor), -1)
        _fake_fd_loss = tf.reduce_mean(_fake_fd_error)

        self.dsurprisal_dinput = tf.gradients(_fake_fd_loss, self.norm_frame)
        self.dsurprisal_dlastinput = tf.gradients(_fake_fd_loss, self.norm_tail_frame)
        return forward_dynamic_error, forward_dynamic_loss
    
    def attach_action_reward_head(self, policy_convnet):
        #(frame_ph) -> action_scores
        #(frame_ph) -> predicted_reward
        #(frame_ph, action_taken_ph, advantage_ph, old_action_probability_ph) -> pg_loss
        #(frame_ph, reward_ph) -> reward_loss
        #(frame_ph) -> entropy_loss
        x = Dense(512, activation=tf.nn.relu)(policy_convnet)
        x = Dense(512, activation=tf.nn.relu)(x)
        
        #poligy gradient loss
        action_scores = Dense(self.num_actions, activation=None, name="action_scores")(x)
        action_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_scores, labels=self.action_taken_ph, name="action_probability")
        ratio = tf.exp(self.old_action_probability_ph - action_probability, name="PPO_ratio")
        neg_adv = -self.advantage_ph
        pg_loss1 = neg_adv * ratio
        pg_loss2 = neg_adv * tf.clip_by_value(ratio, 1.0 - 0.1, 1 + 0.1) #PPO paper uses 0.2, but code uses 0.1
        pg_loss_surr = tf.maximum(pg_loss1, pg_loss2)
        pg_loss = tf.reduce_mean(pg_loss_surr, name="PPO_pg_loss")

        #saliency to see how input affects action
        self.daction_dinp = tf.gradients(action_scores, self.norm_frame, self.action_one_hot)

        #entropy loss, var names lifted straight out of baselines
        a0 = action_scores - tf.reduce_max(action_scores, -1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        entropy_loss = (- 0.001) * tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1), name="entropy_loss") #PPO paper uses 0.01, but code uses 0.001

        #reward loss
        predicted_reward = Dense(1, activation=None, name="PPO_predicted_reward")(x)
        reward_loss = 0.5 * tf.reduce_mean(tf.square(predicted_reward - self.reward_ph), name="PPO_reward_loss")
        return action_scores, predicted_reward, pg_loss, reward_loss, entropy_loss

    def attach_random_move(self, action_scores):
        #(frame_ph) -> sample_move_index
        #(frame_ph) -> sample_action_probability
        random_offset = tf.random_uniform(shape=tf.shape(action_scores), dtype=action_scores.dtype)
        sample_move_index = tf.argmax(action_scores - tf.log(-tf.log(random_offset)), axis=-1)
        sample_action_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_scores, labels=sample_move_index)
        return sample_move_index, sample_action_probability
    
    def forward(self, sess, state):
        return sess.run(
            [self.sample_move_index, self.sample_action_probability, self.predicted_reward],
            feed_dict = {
                self.frame_ph: state
            }
        )

    def build_summary(self):
        tf.summary.scalar("forward_dynamic_loss", self.forward_dynamic_loss)
        tf.summary.scalar("inverse_dynamic_loss", self.inverse_dynamic_loss)
        tf.summary.scalar("pg_loss", self.pg_loss)
        tf.summary.scalar("reward_loss", self.reward_loss)
        tf.summary.scalar("entropy_loss", self.entropy_loss)
        tf.summary.scalar("total_loss", self.pg_loss)
        tf.summary.image("frame", self.frame_ph)
        return tf.summary.merge_all()
        
    def get_surprisal(self, sess, states, action_indexes, tail_state):
        #returns surprisal = self.forward_dynamic_error
        return sess.run(
            self.forward_dynamic_error,
            feed_dict = {
                self.frame_ph: states,
                self.action_taken_ph: action_indexes,
                self.tail_frame_ph: tail_state
            }
        )
  
    def train(self, sess, states, action_taken, old_action_probability, tail_states, advantage, reward):
        #(frame_ph, tail_frame_ph, action_taken_ph) -> inverse_dynamic_loss
        #(frame_ph, tail_frame_ph, action_taken_ph) -> forward_dynamic_loss
        #(frame_ph, action_index_ph, advantage_ph, old_action_probability_ph) -> pg_loss
        #(frame_ph, reward_ph) -> reward_loss
        #(frame_ph) -> entropy_loss
        _, summary = sess.run(
            [self.train_op, self.merged_summary],
            feed_dict = {
                self.frame_ph: states,
                self.tail_frame_ph: tail_states,
                self.action_taken_ph: action_taken,
                self.advantage_ph: advantage,
                self.old_action_probability_ph: old_action_probability,
                self.reward_ph: reward
            }
        )
        return summary
        #hence
        #(frame_ph, action_taken_ph, old_action_probability_ph, tail_states_ph, advantage_ph, reward_ph) -> total_loss

    def get_dsurprisal_dinps(self, sess, states, action_indexes, prev_rollout_last_state):
        _states = states[:-1]
        _tail_state = states[-1]
        if prev_rollout_last_state[0] is None:
            _states = np.insert(_states, 0, _states[0], axis=0)
        else:
            _states = np.insert(_states, 0, prev_rollout_last_state, axis=0)

        grads = sess.run(
            self.dsurprisal_dinput,
            feed_dict = {
                self.frame_ph: _states,
                self.action_taken_ph: action_indexes,
                self.tail_frame_ph: [_tail_state]
            }
        )[0]

        tail_grad =  sess.run(
            self.dsurprisal_dlastinput,
            feed_dict = {
                self.frame_ph: [_states[-1]],
                self.action_taken_ph: [action_indexes[-1]],
                self.tail_frame_ph: [_tail_state]
            }
        )[0]
        return np.append(grads[1:], [tail_grad[0]], axis=0)
    
    def get_daction_dinps(self, sess, states, action_indexes):
        return sess.run(
            self.daction_dinp,
            feed_dict = {
                self.frame_ph: states,
                self.action_taken_ph: action_indexes
            }
        )[0]
