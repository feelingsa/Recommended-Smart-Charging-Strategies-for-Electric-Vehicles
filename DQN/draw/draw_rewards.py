import pickle

file = open("../his_para/dqn_1.pkl", "rb")
dqn_data = pickle.load(file)
print(dqn_data)