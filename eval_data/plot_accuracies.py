from matplotlib import pyplot as plt
import json


data = json.load(open("test_accuracy.json"))


plt.figure()

plt.plot(
    data["reconstruction_weights"], data["accuracies"], label=r"Accuracy($\lambda$)"
)

plt.xlabel(r"$\lambda$")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.savefig("accuracy.png")
