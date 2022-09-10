from  matplotlib import pyplot as plt

a = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
b = [1.63, 1.59, 1.59, 1.53, 1.47, 1.55, 1.57, 1.51, 1.63, 1.58, 1.61, 1.68, 1.64, 1.66, 1.70, 1.69]

plt.figure()

plt.plot(a, b, color='#00FF00')
plt.scatter(a, b, color='#32CD32')

plt.ylabel('Average waiting time ms')
plt.xlabel('n')

plt.show()