# Reconstruction Auxiliary loss

### How to run
Update the paths in run.sh then:
```
python -m pip install -r requirements.txt
cd audio
bash run.sh
```

### Description

#### Unsupervised Reconstruction Loss as a Form of Regularization.

Deep Neural Networks have a big overfitting issue, especially when applied to a
small amount of labeled data. Researchers have devised multiple methods to deal
with the issue, like L1/L2 weight regularization, Dropout, Transfer Learning,
and multi-Task Learning.

In this project, we will focus on using multi-Task learning as a way to improve
the generalization performance of neural networks. The ideas implemented here
are inspired by three really interesting papers:

* [Multitask Learning. Autonomous Agents and Multi-Agent
Systems](http://www.cs.cornell.edu/~caruana/mlj97.pdf)
* [An Overview of Multi-Task Learning in Deep Neural
Networks](https://ruder.io/multi-task/)
* [Supervised autoencoders: Improving generalization performance with unsupervised
regularizers](https://papers.nips.cc/paper/7296-supervised-autoencoders-improving-generalization-performance-with-unsupervised-regularizers.pdf)

The first two papers try to explain why multi-task learning can improve the
performance of individual tasks, some of the possible explanations they provide
are:

**Representation Bias:**

![](https://cdn-images-1.medium.com/max/800/1*mPNY_0kq0Mj4KH39Qbt_XQ.png)

<span class="figcaption_hack">— Image by author</span>

If we train a network on Task T and T’ at the same time, the network becomes
biased towards representations that serve both tasks. This makes the network
most likely to generalize to new tasks.

**Regularization:**

Using multi-Task Learning makes the network less likely to overfit on the noise
from the training data since it reduces the number of possible solutions given
that a solution in MTL needs to work for all tasks at the same time.

**Attention Focusing:**

Training on multiple related tasks can give the model a stronger signal on what
are the relevant features and what is just noise.

The third paper considers a scenario where there is only have one supervised
Task T, so the authors add a new artificial and unsupervised task of
reconstructing the Input. They prove in a simplified setting that adding the
reconstruction loss improves the generalization performance of the supervised
task and show some empirical results that support their hypothesis.

### The Data:

We use the Free Music Archive (FMA) small version in the next experiments. It is
a dataset of 8000 song snippets classified into 8 genres:

    {
        "International": 0,
        "Pop": 1,
        "Instrumental": 2,
        "Hip-Hop": 3,
        "Electronic": 4,
        "Experimental": 5,
        "Folk": 6,
        "Rock": 7
    }

We split the dataset into Train-Val-Test at a ratio of 70%-10%-20% and transform
the raw audio waveform into Mel Spectrograms before feeding them to the network.
For more details about the pre-processing you can look into one of my previous
projects:

[Music Genre Classification: Transformers vs Recurrent Neural
Networks](https://towardsdatascience.com/music-genre-classification-transformers-vs-recurrent-neural-networks-631751a71c58)

![](https://cdn-images-1.medium.com/max/800/1*9sKRBtLy7na60czoPdCC7g.png)
<span class="figcaption_hack">Example of Mel-Spectrogram — Image by author</span>

### The Model:

We apply an LSTM Based neural network along the time axis to classify the music
genre.

![](https://cdn-images-1.medium.com/max/800/1*IeoMkPQiRa3SYHPp-Fg11w.png)

<span class="figcaption_hack">Music Genre model — Image by author</span>

Adding Dropout layers acts as additional regularization and makes the
reconstruction task a little more challenging for the model.<br> We use Pytorch
Lightning to implement this model, the forward function looks like this:

    forward(self, x):
        x = self.do(x)
        x, _ = self.lstm1(x)
        x_seq, _ = self.lstm2(x)
        x, _ = torch.max(self.do(x_seq), dim=1)
        x = F.relu(self.do(self.fc1(x)))
        y_hat = self.fy(x)
        x_reconstruction = torch.clamp(self.fc2(self.do(x_seq)), -1.0, 1.0)
        
    y_hat, x_reconstruction

Now we define the loss as a weighted sum between the classification loss and
reconstruction loss as follows:

> loss = loss_classification + λ * loss_reconstruction

Where λ is a hyper-parameter that helps mitigate the fact that the two losses do
not have the same scale while also giving more control on how much importance we
want to give to the auxiliary task. The loss is defined as follow:

    training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, x_reconstruction = self(x)
        loss_y = F.cross_entropy(y_hat, y)
        loss_x = F.l1_loss(x, x_reconstruction)
        
    loss_y + self.reconstruction_weight * loss_x

### Results:

In the experiments we try out multiple values of Lambda to see which one works
better, the baseline being Lambda = 0, meaning that the auxiliary loss is
ignored.

![](https://cdn-images-1.medium.com/max/800/1*awoDGHwr_E1jW-WeSmJgtQ.png)

We can see that adding the reconstruction loss (λ = 10 and λ = 2) yields better
performance compared to the baseline.

Now in terms of classification accuracy we have:

![](https://cdn-images-1.medium.com/max/800/1*BmjbFW4ELWqTAGayRyauzw.png)

* **Random Guess:** 12.5%
* **Baseline:** Accuracy = 47.5%
* **λ =10**: Accuracy = **51%**

Adding the reconstruction loss gives ~ 3% accuracy improvement over the baseline
while using the same classification architecture.

### Conclusion:

In this project, we showed that adding an auxiliary unsupervised task to a
neural network can improve its generalization performance by acting as an
additional form of regularization. The method to add the reconstruction loss is
easily implemented in Pytorch Lightning but comes at the cost of a new
hyper-parameter λ that we need to optimize.
