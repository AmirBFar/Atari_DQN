import gym
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import threading
import time
from random import choices,uniform,randint
from PIL import Image
import sys

class DQN_Agent():
    
    def __init__(self,env,num_episodes,batch_size,sampling_policy,dir_name,rank):
        self.env = gym.make(env)
        self.RB = ReplayBuffer()
        self.episode_length_vec = []
        self.episode_reward_vec = []
        self.num_episodes = num_episodes
        self.q_net = self.network()
        self.target_q_net = self.network()
        self.batch_size = batch_size
        self.episode_cnt = 0
        self.sampling_policy = sampling_policy
        self.max_num_steps_per_episode = 10000

        self.dir_name = dir_name
        self.rank = rank

        actor_thread = threading.Thread(target=self.actor)
        learner_thread = threading.Thread(target=self.learner)
        actor_thread.start()
        learner_thread.start()

        actor_thread.join()
        learner_thread.join()

    def network(self):
        
        in_obs = tf.keras.layers.Input(shape=[84,84,4])
        #in_is_weights = tf.keras.layers.Input(shape=(1,))
        #in_actual     = tf.keras.layers.Input(shape=(self.env.action_space.n,))
        norm = tf.keras.layers.Lambda(lambda x: x / 255.0)(in_obs)

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,activation="relu", data_format="channels_last", padding="same")(norm)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,activation="relu", data_format="channels_last", padding="same")(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,activation="relu", data_format="channels_last", padding="same")(conv2)
        
        conv_flattened = keras.layers.Flatten()(conv3)
        hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
        output = keras.layers.Dense(self.env.action_space.n,activation="linear")(hidden) ### Q-value prediction
        
        
        model = keras.models.Model(inputs=in_obs, outputs=output)
        optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')

        return model

    def policy(self,state,eps):
        if uniform(0,1) < eps:
            return randint(0,1)
        return self.greedy_policy(state)
        
    def greedy_policy(self,state):
        state_input = tf.convert_to_tensor(state[None,:])
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def learner(self):
        time.sleep(1)
        update_cnt = 0
        while self.episode_cnt < self.num_episodes:
            update_cnt += 1
            batch = self.RB.sample_batch(self.batch_size,self.sampling_policy)
            loss = self.train(batch)
            if update_cnt%10000 == 0:
                self.target_q_net.set_weights(self.q_net.get_weights())
            
            
            
    def train(self,batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q[i][action_batch[i]] = reward_batch[i] if done_batch[i] else reward_batch[i] + 0.99 * max_next_q[i]
        result = self.q_net.fit(x=state_batch, y=target_q,verbose=1)
        return result.history['loss']

    def actor(self):
        state = self.env.reset()
        state = ImageProcess(state)
        state = np.stack([state] * 4, axis = 2)
        episode_length = 0
        episode_reward = 0
        cnt = 0
        episode_step_cnt = 0
        eps = 1
        while self.episode_cnt < self.num_episodes:
            cnt += 1
            action = self.policy(state,eps)
            next_state, reward, done, _ = self.env.step(action)
            
            next_state = ImageProcess(next_state)
            next_state = np.append(state[:, :, 1: ], np.expand_dims(next_state, 2), axis = 2)
            
            self.RB.add([state,action,reward,next_state,done,cnt])
            episode_length += 1
            episode_reward += reward

            if done or episode_step_cnt == self.max_num_steps_per_episode:
                with open("%s/%d.txt"%(self.dir_name,self.rank), 'a+') as f1:
                    f1.write("%.2f,%.2f\n"%(episode_reward,episode_length))
                state = self.env.reset()
                state = ImageProcess(state)
                state = np.stack([state] * 4, axis = 2)
                #self.episode_length_vec.append(episode_length)
                #self.episode_reward_vec.append(episode_reward)
                if self.episode_cnt%10 == 0:
                    print("Episode {} finished, with length {}".format(self.episode_cnt,episode_length))
                    print("Total number of steps is {}".format(cnt))
                    sys.stdout.flush()
                self.episode_cnt += 1
                episode_length = 0
                episode_reward = 0
                episode_step_cnt = 0
                if eps > .1:
                    eps = eps - 1/self.num_episodes
            else:
                state = next_state


class ReplayBuffer():

    def __init__(self):
        self.replay_buffer = deque(maxlen=int(1e5))
        
    def add(self,exp):
        self.replay_buffer.append(exp)
        
    def sample_batch(self,batch_size,sampling_policy):
        if sampling_policy == "Uniform":
            batch = choices(self.replay_buffer,k=batch_size)
        elif sampling_policy == "PER":
            temp_buffer = list(self.replay_buffer)
            weights = [x[-1] for x in list(temp_buffer)]
            batch = choices(temp_buffer,weights=weights,k=batch_size)
        return [np.array([x[i] for x in batch]) for i in range(5)]



def ImageProcess(state):
    state = tf.image.rgb_to_grayscale(state)
    state = tf.image.crop_to_bounding_box(state, 34, 0, 160, 160)
    state = tf.image.resize(state, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.squeeze(state)
        


if __name__ == "__main__":
    dqn = DQN_Agent("BreakoutDeterministic-v4",20000,32,"Uniform")
    print(dqn.episode_length_vec)
