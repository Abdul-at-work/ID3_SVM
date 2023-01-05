import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))

config1=SVC(kernel='rbf',C=1).fit(dataset,labels)
config2=SVC(kernel='rbf',C=10).fit(dataset,labels)
config3=SVC(kernel='poly',C=1).fit(dataset,labels)
config4=SVC(kernel='poly',C=10).fit(dataset,labels)

f, ax = plt.subplots(2, 2, figsize=(10, 15))
display = DecisionBoundaryDisplay.from_estimator(config1, dataset, ax=ax[0][0],plot_method='contour',response_method="predict")
display.ax_.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k')
ax[0][0].set_title(f"Kernel: rbf, C: 1")
display = DecisionBoundaryDisplay.from_estimator(config2, dataset, ax=ax[0][1], plot_method='contour',response_method="predict")
display.ax_.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k')
ax[0][1].set_title(f"Kernel: rbf, C: 10")
display = DecisionBoundaryDisplay.from_estimator(config3, dataset, ax=ax[1][0], alpha=0.7,response_method="predict")
display.ax_.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k')
ax[1][0].set_title(f"Kernel: poly, C: 1")
display = DecisionBoundaryDisplay.from_estimator(config4, dataset, ax=ax[1][1], alpha=0.7,response_method="predict")
display.ax_.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k')
ax[1][1].set_title(f"Kernel: poly, C: 10")
plt.show()
