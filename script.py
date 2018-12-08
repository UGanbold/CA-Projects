import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits.target[100])
plt.gray()
plt.matshow(digits.images[100])
plt.show()
model = KMeans(n_clusters = 10, random_state = 43)
model.fit(digits.data)
fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center Images', fontsize=12, fontweight='bold')
for i in range(10):
  ax=fig.add_subplot(2,5,1+i)
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
plt.show()
new_samples=np.array([
[0.00,0.15,4.10,6.56,6.86,6.38,0.82,0.00,0.07,5.47,7.62,5.37,4.11,7.63,3.87,0.00,2.36,7.62,3.87,0.00,0.00,5.95,6.76,0.08,3.81,7.32,0.23,0.00,0.00,3.13,7.62,0.76,3.74,7.47,0.46,0.00,0.00,2.52,7.62,0.76,2.14,7.62,3.84,0.00,0.18,5.01,7.47,0.30,0.00,5.77,7.50,5.18,6.46,7.62,4.53,0.00,0.00,0.92,5.03,6.10,5.72,3.31,0.05,0.00],
[0.00,0.79,6.89,7.62,7.62,4.23,0.00,0.00,0.00,4.42,7.55,2.85,4.93,7.62,2.04,0.00,0.53,7.17,5.19,0.00,0.59,7.32,5.19,0.00,3.05,7.62,1.91,0.00,0.00,4.42,7.47,0.38,4.12,7.17,0.00,0.00,0.00,2.21,7.62,1.53,4.57,6.86,0.13,0.00,0.00,2.21,7.62,1.53,3.18,7.62,5.01,2.36,1.75,6.00,7.24,0.30,0.08,4.60,7.55,7.62,7.62,7.60,3.23,0.00],
[0.00,0.87,4.42,4.57,4.57,4.57,2.34,0.00,0.00,5.01,7.55,6.10,6.10,6.10,3.26,0.00,0.00,6.71,5.01,0.00,0.00,0.00,0.00,0.00,0.00,6.79,6.79,5.34,5.34,3.00,0.00,0.00,0.00,2.16,5.26,5.34,6.56,7.55,1.68,0.00,0.00,0.31,0.30,0.00,0.79,7.63,3.81,0.00,0.00,5.11,7.40,6.86,7.17,7.62,3.16,0.00,0.00,1.50,3.81,3.81,3.81,2.64,0.00,0.00],
[0.00,4.17,7.62,7.62,7.62,7.40,3.03,0.00,0.00,3.89,7.17,3.05,3.13,3.81,1.50,0.00,0.00,4.65,7.24,4.95,5.34,4.35,0.62,0.00,0.00,4.99,7.62,6.94,5.79,7.60,5.06,0.00,0.00,0.23,0.69,0.00,0.00,5.22,6.10,0.00,0.23,4.35,2.04,2.21,3.05,6.81,5.95,0.00,0.38,7.28,7.62,7.62,7.62,6.86,2.14,0.00,0.00,0.89,1.52,1.37,0.76,0.00,0.00,0.00]
])
new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] ==0:
    print(4, end='')
  elif new_labels[i] == 1:
    print(1, end='')
  elif new_labels[i] == 2:
    print(7, end='')
  elif new_labels[i] == 3:
    print(9, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(3, end='')
  elif new_labels[i] == 6:
    print(0, end='')
  elif new_labels[i] == 7:
    print(8, end='')
  elif new_labels[i] == 8:
    print(5, end='')
  elif new_labels[i] == 9:
    print(2, end='')
print(new_labels)
