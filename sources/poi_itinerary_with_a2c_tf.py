# A2C with TensorFlow/Keras for POI itinerary planning

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from datetime import datetime, timedelta
from PoiEnv import poi_env
from utils import neural_poi_map

class Actor(Model):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)

class Critic(Model):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)

def train_a2c(env, actor, critic, optimizer_actor, optimizer_critic, poi_start, time_input, date_input, episodes=10, gamma=0.99):
    map_from_poi_to_action, map_from_action_to_poi = neural_poi_map()
    best_path = []
    best_reward = -float("inf")

    for ep in range(episodes):
        state = env.reset(poi_start, timedelta(hours=time_input), date_input)
        done = False
        log_probs = []
        values = []
        rewards = []
        visited_poi = []
        valid_step_count = 0

        while not done:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            probs = actor(state_tensor).numpy()[0]

            valid_actions = list(env.action_space)
            action_probs = np.array([probs[map_from_poi_to_action[a]] if a in env.action_space else 0 for a in map_from_action_to_poi.values()])

            if action_probs.sum() == 0:
                break
            action_probs /= action_probs.sum()

            action_index = np.random.choice(len(action_probs), p=action_probs)
            action = map_from_action_to_poi[action_index]

            if action not in env.action_space:
                continue

            visited_poi.append(action)
            value = critic(state_tensor)[0, 0].numpy()

            new_state, reward, done = env.step(action)

            log_probs.append(np.log(action_probs[action_index] + 1e-8))
            values.append(value)
            rewards.append(reward)
            state = new_state
            valid_step_count += 1

        if valid_step_count < 2 or len(log_probs) == 0 or len(rewards) == 0:
            continue

        if sum(rewards) > best_reward:
            best_reward = sum(rewards)
            best_path = visited_poi.copy()

        values.append(0)
        Qvals = []
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * values[t + 1]
            Qvals.insert(0, Qval)

        values = tf.convert_to_tensor(values[:-1], dtype=tf.float32)
        Qvals = tf.convert_to_tensor(Qvals, dtype=tf.float32)
        log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

        advantage = Qvals - values

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actor_loss = -tf.reduce_mean(log_probs * advantage)
            critic_pred = critic(tf.convert_to_tensor([state], dtype=tf.float32))
            critic_loss = tf.reduce_mean(tf.square(advantage))

        grads1 = tape1.gradient(actor_loss, actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, critic.trainable_variables)

        if all(g is None for g in grads1 + grads2):
            continue

        optimizer_actor.apply_gradients(zip(grads1, actor.trainable_variables))
        optimizer_critic.apply_gradients(zip(grads2, critic.trainable_variables))

    return best_path, best_reward

def evaluate_a2c(df_poi_it, df_crowding, df_poi_time_travel, df_weather, group_data):
    env = poi_env(df_poi_it, df_crowding, df_poi_time_travel)
    map_from_poi_to_action, _ = neural_poi_map()

    input_dim = len(df_poi_it.id) + 2
    output_dim = len(df_poi_it.id)
    actor = Actor(input_dim, output_dim)
    critic = Critic(input_dim)
    optimizer_actor = optimizers.Adam(learning_rate=1e-3)
    optimizer_critic = optimizers.Adam(learning_rate=1e-3)

    poi_start = group_data['poi'].iloc[0]
    date_input = datetime.strptime(group_data['data_visita'].iloc[0], '%Y-%m-%d')
    first_visit_time = datetime.strptime(group_data['ora_visita'].iloc[0], '%H:%M:%S')
    last_visit_time = datetime.strptime(group_data['ora_visita'].iloc[-1], '%H:%M:%S')
    last_poi = group_data['poi'].iloc[-1]
    time_input = (last_visit_time + timedelta(minutes=env.poi_time_visit[last_poi])).hour - first_visit_time.hour

    best_path, best_reward = train_a2c(env, actor, critic, optimizer_actor, optimizer_critic, poi_start, time_input, date_input)

    total_time_visit, total_time_distance, total_time_crowd, time_left = env.time_stats()
    popular_poi = df_poi_it.sort_values(by='Time_Visit', ascending=False)['id'].values[:3]
    popular_poi_visited = len(set(best_path) & set(popular_poi))

    return {
        'itinerary': best_path,
        'reward': best_reward,
        'time_visit': total_time_visit,
        'time_distance': total_time_distance,
        'time_crowd': total_time_crowd,
        'time_left': time_left,
        'popular_poi_visited': popular_poi_visited,
        'poi_len': len(best_path)
    }
