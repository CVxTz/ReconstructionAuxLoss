from pandas import read_csv
from seaborn import lineplot
from matplotlib import pyplot as plt
import json


data = json.load(open('test_accuracy.json'))


plt.figure()

plt.plot(data['reconstruction_weights'], data['accuracies'], label="Accuracy($\lambda$)")

plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
plt.xscale("log")

plt.show()