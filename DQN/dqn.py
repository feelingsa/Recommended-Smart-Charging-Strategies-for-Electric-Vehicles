# -*- coding: utf-8 -*-
# import the necessary packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from dqn_env import parallel_env

# 1. Define some Hyper Parameters
BATCH_SIZE = 32  # batch size of sampling process from buffer
LR = 0.001  # learning rate
EPSILON = 0.9  # epsilon used for epsilon greedy approach 用于epsilon贪婪方法的epsilon
GAMMA = 0.9  # discount factor 折扣系数
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates 目标网路更新的频率如何
MEMORY_CAPACITY = 50000  # The capacity of experience replay buffer 经验回放缓冲器的容量

# env = gym.make("CartPole-v0")  # Use cartpole game as environment
# env = env.unwrapped
env = parallel_env()
# N_ACTIONS = env.action_space[0].n  # 10 actions
# N_STATES = env.observation_space[0].shape[0]  # 10 states
# print(env.observation_space)
# ENV_A_SHAPE = 0 if isinstance(env.action_space[0].sample(),
#                               int) else env.action_space[0].sample().shape  # to confirm the shape
# -----!-----

N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0] # 4 states
print(N_STATES)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape

# 2. Define the network used in both target net and the net for training
class Net(nn.Module):
    def __init__(self):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
        self.out = nn.Linear(10, N_ACTIONS)  # layer 2
        self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# 3. Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = Net(), Net()
        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process 计算学习过程的步骤
        self.memory_counter = 0  # counter used for experience replay buffer 用于经验回放缓冲区的计数器

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        # print('transition:', transition)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step






'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
dqn = DQN()
episode_his = []
reward_his = []
load_his = []
wait_his = []
dis_his = []
episode_reward_his = 0
step = 0
# Start training
print("\nCollecting experience...")
ep_r = 0
load = 0
wt = 0
dis = 0
for i_episode in range(2000):
    # play 400 episodes of cartpole game
    s = env.reset()
    while True:
        env.render()
        # take action based on the current state
        a = dqn.choose_action(s)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)
        Load, waiting_time, Dis = env.history()
        # store the transitions of states
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        load += Load
        dis += Dis
        # print('waitingtime:', waiting_time)
        wt += waiting_time
        # print('wt:', wt)
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            # if done:

        if done:
            # if game is over, then skip the while loop.
            if i_episode % 10 == 0:
                # print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r/step, 5))
                # print('wt:', round(wt, 10))
                # if ep_r < -150:
                #     ep_r = reward_his[-1]
                episode_his.append(i_episode)
                reward_his.append(ep_r / step)
                load_his.append(load / step)
                wait_his.append(wt / step)
                dis_his.append(dis / step)
                step = 0
                ep_r = 0
                load = 0
                wt = 0
                dis = 0
            break
        # use next state to update the current state.
        s = s_
        step += 1
        torch.save(dqn.target_net.state_dict(), './result/DQN_n_9.pth')



# draw
from matplotlib import pyplot as plt
import pickle
# print(reward_his)
# save


def save_his(episode_his, reward_his, wait_his, load_his, dis_his, act_his, save_r_name, save_w_name, save_l_name, save_d_name, save_a_name):
    R_his_save = []
    W_his_save = []
    L_his_save = []
    D_his_save = []
    A_his_save = []
    for i in range(len(episode_his)):
        R_his_save.append([episode_his[i], reward_his[i]])
        W_his_save.append([episode_his[i], wait_his[i]])
        L_his_save.append([episode_his[i], load_his[i]])
        D_his_save.append([episode_his[i], dis_his[i]])
        A_his_save.append([episode_his[i], act_his[i]])
    f1 = open('his_para/n/test5/' + save_r_name + '.pkl', 'wb')
    pickle.dump(R_his_save, f1)
    f1.close()

    f2 = open('his_para/n/test5/' + save_w_name + '.pkl', 'wb')
    pickle.dump(W_his_save, f2)
    f2.close()

    f3 = open('his_para/n/test5/' + save_l_name + '.pkl', 'wb')
    pickle.dump(L_his_save, f3)
    f3.close()

    f4 = open('his_para/n/test5/' + save_d_name + '.pkl', 'wb')
    pickle.dump(D_his_save, f4)
    f4.close()

    f5 = open('his_para/n/test5/' + save_a_name + '.pkl', 'wb')
    pickle.dump(A_his_save, f5)
    f5.close()

    plt.figure(dpi=100)
    plt.title('Reward')
    plt.plot(episode_his, reward_his)
    plt.ylabel('Reward')
    plt.xlabel('Steps')

    plt.figure(dpi=100)
    plt.title('Load')
    plt.plot(episode_his, load_his)
    plt.ylabel('Load')
    plt.xlabel('Steps')

    plt.figure(dpi=100)
    plt.title('Waiting_time')
    plt.plot(episode_his, wait_his)
    plt.ylabel('Wait_time ms')
    plt.xlabel('Steps')

    plt.figure(dpi=100)
    plt.title('Distance')
    plt.plot(episode_his, dis_his)
    plt.ylabel('Dis')
    plt.xlabel('Steps')

    plt.show()

act_his = dis_his
save_his(episode_his, reward_his, wait_his, load_his, dis_his, act_his, 'dqn_Reward_test', 'dqn_WaitingTime_test', 'dqn_Load_test', 'dqn_Distance_test', 'dqn_Action_test')
# R_his_save = []
# W_his_save = []
# L_his_save = []
# for i in range(len(episode_his)):
#     R_his_save.append([episode_his[i], reward_his[i]])
#     W_his_save.append([episode_his[i], wait_his[i]])
#     L_his_save.append([episode_his[i], load_his[i]])
# f1 = open('his_para/test/dqn_Reward_10000.pkl', 'wb')
# pickle.dump(R_his_save, f1)
# f1.close()
#
# f2 = open('his_para/test/dqn_WaitingTime_10000.pkl', 'wb')
# pickle.dump(W_his_save, f2)
# f2.close()
#
# f3 = open('his_para/test/.pkl', 'wb')
# pickle.dump(L_his_save, f3)
# f3.close()
#
# plt.figure(dpi=100)
# plt.title('Reward')
# plt.plot(episode_his, reward_his)
#
# plt.figure(dpi=100)
# plt.title('Load')
# plt.plot(episode_his, load_his)
#
# plt.figure(dpi=100)
# plt.title('Waiting_time')
# plt.plot(episode_his, wait_his)
# plt.show()




# test
print('...Start Testing...')
# test model----------------------

test_done = False
a = test_done
test = True
env.test = test


s = env.reset()
env.render()
n_r = env.n_r
inf_c = env.inf_c
r_n_c = env.r_n_c
l_c = env.l_c
l_q = env.l_q
s_q = env.s_q
s_c = env.s_c
a_t = env.a_t
c_w_l = env.c_w_l
STEP = env.STEP
Load = env.Load
waiting = env.waiting
print('---------reset_info----------')

step_list = []
R_list = []
L_list = []
W_list = []
D_list = []
ep_r = 0
steps = 0
A_list = []
while True:
    env.render()
    steps += 1
    # take action based on the current state
    a = dqn.choose_action(s)
    # obtain the reward and next state and some other information
    s_, r, done, info = env.step(a)
    dqn.store_transition(s, a, r, s_)

    l, w, d = env.history()
    step_list.append(steps)
    A_list.append(a)
    R_list.append(r)
    L_list.append(-l)
    W_list.append(w)
    D_list.append(d)

    if done:
        # if game is over, then skip the while loop.
        break
    # use next state to update the current state.
    s = s_
    # print('steps:', steps)

print('...End Testing...')

print('...Start Saving...')
save_his(step_list, R_list, W_list, L_list, D_list, A_list, 'Test_Reward', 'Test_Wait_time', 'Test_Load', 'Test_Distance', 'Test_Action')


print('---------Test nearest----------')
env.n_r = n_r
env.inf_c = inf_c
env.r_n_c = r_n_c
env.l_c = l_c
env.l_q = l_q
env.s_q = s_q
env.s_c = s_c
env.a_t = a_t
env.c_w_l = c_w_l
env.STEP = STEP
env.Load = Load
env.waiting = waiting

step_list = []
R_list = []
L_list = []
W_list = []
D_list = []
A_list = []
ep_r = 0
steps = 0
while True:
    env.render()
    steps += 1
    # print(env.l_q)
    # take action based on the current state
    a = env.choose_nearest()
    # obtain the reward and next state and some other information
    s_, r, done, info = env.step(a)

    l, w, d = env.history()
    step_list.append(steps)
    R_list.append(r)
    L_list.append(-l)
    W_list.append(w)
    D_list.append(d)
    A_list.append(a)

    if done:
        # if game is over, then skip the while loop.
        break
    # use next state to update the current state.
    s = s_

save_his(step_list, R_list, W_list, L_list, D_list, A_list, 'Nearest_Reward', 'Nearest_Wait_time', 'Nearest_Load', 'Nearest_Distance', 'Nearest_Action')


