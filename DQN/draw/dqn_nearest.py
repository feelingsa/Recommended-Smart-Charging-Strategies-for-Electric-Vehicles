import pickle
from matplotlib import pyplot as plt
import numpy as np

# ------------------------------------WT-----------------------------------------
file1 = open("../his_para/n/test5/Test_Wait_time.pkl", "rb")
dqn_data = pickle.load(file1)
# print(dqn_data)

DQN_W_His = []
Nearest_W_His = []
STEP_His = []
D_w = 0
N_w = 0
t = 0
for item in dqn_data:
    D_w += item[1]
    DQN_W_His.append(item[1])
    STEP_His.append(item[0])

file2 = open("../his_para/n/test5/Nearest_Wait_time.pkl", "rb")
nearest_data = pickle.load(file2)

for item in nearest_data:
    N_w += item[1]
    Nearest_W_His.append(item[1])


print('d', D_w/len(dqn_data))
print('n', N_w/len(nearest_data))
plt.figure()
# plt.title('DQN algorithm wait time diagram')
plt.title('Scenario 3 Waiting Time Chart')
# ax1 = plt.subplot(121)
plt.plot(STEP_His, DQN_W_His, color='#00FF00')
# ax1.title('DQN')
plt.ylabel('Wait time ms')
plt.xlabel('Steps')
plt.figure()
plt.title('nearest algorithm wait time diagram')
# ax2 = plt.subplot(122)
plt.plot(STEP_His, Nearest_W_His, color='#FF8C00')
# ax2.title('Nearest')
plt.ylabel('Wait time ms')
plt.xlabel('Steps')
plt.show()

# ------------------------------------L-----------------------------------
# file1 = open("../his_para/test1/Test_Load.pkl", "rb")
# dqn_data = pickle.load(file1)
# # print(dqn_data)
#
# DQN_W_His = []
# Nearest_W_His = []
# STEP_His = []
# D_w = 0
# N_w = 0
# t = 0
# for item in dqn_data:
#     D_w += item[1]
#     DQN_W_His.append(item[1])
#     STEP_His.append(item[0])
#
# file2 = open("../his_para/test1/Nearest_Load.pkl", "rb")
# nearest_data = pickle.load(file2)
#
# for item in nearest_data:
#     N_w += item[1]
#     Nearest_W_His.append(item[1])
#
#
# print('d', D_w/len(dqn_data))
# print('n', N_w/len(nearest_data))
# plt.figure()
# # plt.title('DQN algorithm load diagram')
# plt.title('Scenario 2 Load Chart')
# # ax1 = plt.subplot(121)
# plt.plot(STEP_His, DQN_W_His, color='#FF1493')
# # ax1.title('DQN')
# plt.ylabel('Load')
# plt.xlabel('Steps')
# plt.figure()
# plt.title('Shortest path algorithm load diagram')
# # ax2 = plt.subplot(122)
# plt.plot(STEP_His, Nearest_W_His, color='#FF8C00')
# # ax2.title('Nearest')
# plt.ylabel('Load')
# plt.xlabel('Steps')
# plt.show()

# ----------------------------------D------------------------------------
# file1 = open("../his_para/test1/Test_Distance.pkl", "rb")
# dqn_data = pickle.load(file1)
# # print(dqn_data)
#
# DQN_W_His = []
# Nearest_W_His = []
# STEP_His = []
# D_w = 0
# N_w = 0
# t = 0
# for item in dqn_data:
#     D_w += item[1]
#     DQN_W_His.append(item[1])
#     STEP_His.append(item[0])
#
# file2 = open("../his_para/test1/Nearest_Distance.pkl", "rb")
# nearest_data = pickle.load(file2)
#
# for item in nearest_data:
#     N_w += item[1]
#     Nearest_W_His.append(item[1])
#
# a = 0
# for i in range(len(dqn_data)):
#     if dqn_data[i][1] == nearest_data[i][1]:
#         a += 1
#         print(i, dqn_data[i][1])
# print(a)
# plt.figure()
# plt.title('DQN algorithm distance diagram')
# plt.scatter(STEP_His, DQN_W_His, color='#4169E1')
#
# # ax2.title('Nearest')
# plt.ylabel('Distance')
# plt.xlabel('Steps')
#
# plt.figure()
# plt.title('Nearest algorithm distance diagram')
# plt.scatter(STEP_His, Nearest_W_His, color='#FF8C00')
# plt.ylabel('Distance')
# plt.xlabel('Steps')
# plt.show()