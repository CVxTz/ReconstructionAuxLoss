import pandas as pd
from matplotlib import pyplot as plt


df0 = pd.read_csv("run-Lambda=0.0-tag-valid_loss_y.csv")
df10 = pd.read_csv("run-Lambda=10.0-tag-valid_loss_y.csv")


plt.figure()

plt.plot(df0["Step"], df0["Value"], label="Baseline")
plt.plot(df10["Step"], df10["Value"], label=r"$\lambda$ = 10")

plt.xlabel("Step")
plt.ylabel("Loss (Lower is better)")
plt.legend()
plt.savefig("loss.png")
