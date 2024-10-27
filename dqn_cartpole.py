import gym
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber

env = gym.make("CartPole-v1")

# Online Network creating these datas are given in the documentation
net_input = Input(shape=(4,))
x = Dense(32, activation="relu")(net_input)
x = Dense(16, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

q_net.compile(optimizer="adam")
loss_fn = Huber()

# Target Network cloning
target_net = clone_model(q_net)

# Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.005 #here we are dividing so we take greater tahn one
GAMMA = 0.99
MAX_TRANSITIONS = 1_00_000
NUM_EPISODES = 400
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 4
LEARN_AFTER_STEPS = 3

REPLAY_BUFFER = [] #just  store list of transition


# Inserts transition into Replay Buffer
def insert_transition(transition):
    #to stop and make the memory not full
    if len(REPLAY_BUFFER) >= MAX_TRANSITIONS:
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)


# Samples a batch of transitions from Replay Buffer randomly
def sample_transitions(batch_size=16):
    random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(REPLAY_BUFFER), dtype=tf.int32)
    #now we are storing the below information for every baatch

    sampled_current_states = []
    sampled_actions = []
    sampled_rewards = []
    sampled_next_states = []
    sampled_terminals = []

     #getting current states
    for index in random_indices:
        sampled_current_states.append(REPLAY_BUFFER[index][0])
        sampled_actions.append(REPLAY_BUFFER[index][1])
        sampled_rewards.append(REPLAY_BUFFER[index][2])
        sampled_next_states.append(REPLAY_BUFFER[index][3])
        sampled_terminals.append(REPLAY_BUFFER[index][4])
   #converting into tensors
    return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(sampled_actions), tf.convert_to_tensor(
        sampled_rewards), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)


# Epsilon-Greedy Policy
#we areexpanding the dimensions of the satte because
def policy(state, explore=0.0):
    action = tf.argmax(q_net(tf.expand_dims(state, axis=0))[0], output_type=tf.int32)
    #shape is a tuple
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), maxval=2, dtype=tf.int32)
    return action


# Our Reward function
#pole angle is low and pole angular velocity is low we will reward positively

def calculate_reward(state):
    # -1 Reward for every step
    reward = -1.0

    # Reward +1.0 when,
    # Cart Position is between +-0.5,
    # Cart Velocity is within +-1,
    # Pole Angle is lesser than 4 degrees, i.e is 0.07 and
    # pole angular velocity is lower than 15% of what can be maximum -.535
    if -0.5 <= state[0] <= 0.5 and -1 <= state[1] <= 1 and -0.07 <= state[2] <= 0.07 and -0.525 <= state[3] <= 0.525:
        reward = 1.0

    return reward


# Gathering Random initial states for Average Q Metric
random_states = []
done = False
i = 0
state = env.reset()
while i < 20 and not done:
    random_states.append(state)
    state, _, done, _ = env.step(policy(state).numpy())
    i += 1

random_states = tf.convert_to_tensor(random_states)


def get_q_values(states):
    return tf.reduce_max(q_net(states), axis=1)


# Initializations before training
step_counter = 0
#just like dictionaries
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}

for episode in range(NUM_EPISODES):

    # Deep Q-Learning Algorithm
    done = False
    total_rewards = 0
    episode_length = 0
    state = env.reset()
    while not done:
        action = policy(state, EPSILON)
        next_state, _, done, _ = env.step(action.numpy())
        reward = calculate_reward(next_state)
        insert_transition([state, action, reward, next_state, done])
        state = next_state
        step_counter += 1

        #applying update after a crtain  number of updates
        if step_counter % LEARN_AFTER_STEPS == 0:
            #sampling these from the sample transitions
            current_states, actions, rewards, next_states, terminals = sample_transitions(BATCH_SIZE)

            next_action_values = tf.reduce_max(target_net(next_states), axis=1)
            #reduce max to select optimal value q value
            # where function does  if target is true it takes reards and if dfalse it takes rewards + gamma
            targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

            with tf.GradientTape() as tape:
                preds = q_net(current_states)
                #whats the logic of this code ??
                batch_nums = tf.range(0, limit=BATCH_SIZE)
                indices = tf.stack((batch_nums, actions), axis=1)
                current_values = tf.gather_nd(preds, indices)
               #loss if MSE loss
                loss = loss_fn(targets, current_values)
            grads = tape.gradient(loss, q_net.trainable_weights)
            q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        if step_counter % TARGET_UPDATE_AFTER == 0:
            #updating target networks just copying into taget
            target_net.set_weights(q_net.get_weights())

        total_rewards += reward
        episode_length += 1
        print("Working loop now ", "episode", episode)

    # Saving Metrics
    avg_q = tf.reduce_mean(get_q_values(random_states)).numpy()
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_rewards)
    metric["avg_q"].append(avg_q)
    metric["exploration"].append(EPSILON)
    EPSILON /= EPSILON_DECAY
    #no pickel dump

 #storing in panda a daataframe
    pd.DataFrame(metric).to_csv("metric.csv", index=False)
env.close()
q_net.save("dqn_q_net")


#hubers law
#We need to clip the error term. from the update. R plus Gamma Max Q. to be between minus one and one because the absolute
# value loss function X has a derivative of minus 1 for all negative value of X and a derivative of one for all positive value
# of X clipping the squared error to be minus 1 and 1 corresponds to using an absolute value function for error. outside of the
# minus one, one interval