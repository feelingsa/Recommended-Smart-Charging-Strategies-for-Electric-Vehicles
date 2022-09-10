from  nearest_env import  parallel_env

env = parallel_env()
episode = 2000
episode_his = []
reward_his = []
load_his = []
wait_his = []
episode_reward_his = 0
step = 0


for i_episode in range(episode):
    # play 400 episodes of cartpole game
    s = env.reset()
    ep_r = 0
    load = 0
    while True:
        env.render()
        # take action based on the current state
        a = env.choose_action(s)
        # print(a)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)
        Load, waiting_time = env.history()
        # store the transitions of states

        ep_r += r
        load += Load
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if done:
            if i_episode % 10 == 0:
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r/step, 5))
                    # if ep_r < -150:
                    #     ep_r = reward_his[-1]
                episode_his.append(i_episode)
                reward_his.append(ep_r/step)
                load_his.append(load)
                wait_his.append(waiting_time)
                step = 0
            break
        # use next state to update the current state.
        s = s_
        step += 1

# draw
from matplotlib import pyplot as plt
import pickle
# print(reward_his)
# save
R_his_save = []
W_his_save = []
L_his_save = []
for i in range(len(episode_his)):
    R_his_save.append([episode_his[i], reward_his[i]])
    W_his_save.append([episode_his[i], wait_his[i]])
    L_his_save.append([episode_his[i], load_his[i]])
f1 = open('his_para/nearest_Reward.pkl', 'wb')
pickle.dump(R_his_save, f1)
f1.close()

f2 = open('his_para/nearest_WaitingTime.pkl', 'wb')
pickle.dump(W_his_save, f2)
f2.close()

f3 = open('his_para/nearest_Load.pkl', 'wb')
pickle.dump(L_his_save, f3)
f3.close()

plt.figure(dpi=100)
plt.title('Reward')
plt.plot(episode_his, reward_his)

plt.figure(dpi=100)
plt.title('Load')
plt.plot(episode_his, load_his)

plt.figure(dpi=100)
plt.title('Waiting_time')
plt.plot(episode_his, wait_his)
plt.show()