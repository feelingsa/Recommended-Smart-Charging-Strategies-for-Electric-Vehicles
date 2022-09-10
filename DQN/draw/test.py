import pickle
from matplotlib import pyplot as plt
import numpy as np
file1 = open("../his_para/test3/Test_Action.pkl", "rb")
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

file2 = open("../his_para/test3/Nearest_Action.pkl", "rb")
nearest_data = pickle.load(file2)

for item in nearest_data:
    N_w += item[1]
    Nearest_W_His.append(item[1])

for i in range(len(dqn_data)):
    if dqn_data[i][1] == nearest_data[i][1]:
        print(i, dqn_data[i][1])


plt.figure()
plt.title('DQN algorithm distance diagram')
plt.scatter(STEP_His, DQN_W_His, color='#4169E1')

# ax2.title('Nearest')
plt.ylabel('Distance')
plt.xlabel('Steps')

plt.figure()
plt.title('nearest algorithm distance diagram')
plt.scatter(STEP_His, Nearest_W_His, color='#FF8C00')
plt.ylabel('Distance')
plt.xlabel('Steps')

plt.show()