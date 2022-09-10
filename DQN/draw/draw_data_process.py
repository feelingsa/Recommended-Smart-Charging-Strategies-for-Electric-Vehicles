import pickle
from matplotlib import pyplot as plt
import numpy as np
# ------------------------------------WT--------------------------------------------
# file = open("../his_para/test/dqn_WaitingTime_10000.pkl", "rb")
# dqn_data = pickle.load(file)
# print(len(dqn_data))
# a = 1
# WT = 0
# WT_His = []
# EP_His = []
# for item in dqn_data:
#     if item[0] % 100 == 0:
#         print(item[0], WT/a, a)
#         WT_His.append(WT/a)
#         EP_His.append(item[0])
#         a = 0
#         WT = 0
#     else:
#         a += 1
#         WT += item[1]
# # WT_His[-1] = 2772.932011789746
# # f1 = open('../his_para/nearest_WT_1.pkl', 'wb')
# # pickle.dump(WT_His, f1)
# # f1.close()
# # print(WT_His)
#
# plt.figure()
# plt.title('WT')
# plt.plot(EP_His, WT_His)
# plt.show()


# ------------------------------------------------------Load-----------------------------------------------------------
file = open("../his_para/test/dqn_Load_10000.pkl", "rb")
dqn_data = pickle.load(file)
# print(dqn_data)
a = 1
L = 0
L_His = []
EP_His = []
t = 0
for item in dqn_data:
    if t < 401:
            print(t)
            L_His.append(item[1])
            EP_His.append(item[0])
    t += 1

# L_His[0] = 0.1274009259259259
# f1 = open('../his_para/Nearest_L_1.pkl', 'wb')
# pickle.dump(L_His, f1)
# f1.close()
# print(WT_His)

plt.figure()
plt.title('L')
plt.plot(EP_His, L_His)
plt.show()


# --------------------------------------------------------------------Reward-----------------------------------------------------------
# file = open("../his_para/test/dqn_Reward_10000.pkl", "rb")
# dqn_data = pickle.load(file)
# # print(len(dqn_data))
# a = 1
# WT = 0
# WT_His = []
# EP_His = []
# t = 0
# for item in dqn_data:
#     if t < 401:
#         print(t)
#         WT_His.append(item[1])
#         EP_His.append(item[0])
#     t += 1
#
# print(len(EP_His))
# # WT_His[-1] = 2772.932011789746
# # f1 = open('../his_para/nearest_WT_1.pkl', 'wb')
# # pickle.dump(WT_His, f1)
# # f1.close()
# # print(WT_His)
#
# plt.figure()
# # plt.title('WT')
# plt.plot(EP_His, WT_His)
# plt.ylabel('Reward')
# plt.xlabel('Episode')
# plt.show()