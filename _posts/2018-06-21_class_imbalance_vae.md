---
layout: post
title: Solving Class Imbalance problem using Variational Auto Encoder
author: "Abhishek Mishra"
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<style>
.center-image
{
    margin: 0 auto;
    display: block;
}
</style>

# What is class imbalance?
[code](https://github.com/abhishm/vae_class_imbalance)

Lets assume that you are solving a classification problem involving only two classes. In this problem, there are millions of data from one class and only hundreds of data from the other class. Your goal is given the input, predict which class the input belongs. To solve these kind of problems, the typical steps are as following:
1. Decide a model architecture.
2. Take some random samples from the data.
3. Train the model to reduce the classification loss.
4. Repeat this process until you get good accuracy.

However, the process of random sampling to create a batch will not work when your dataset is highly skewed. It is because each batch will have most of the samples from one class. Your model will be biased. I did an experiment to show the data imbalance problem for a logistic regression model. You can find the more detail of the project [here](https://github.com/abhishm/vae_class_imbalance/blob/master/class-imbalance-experiments.ipynb). There are two main ways to solve this problem:
1. By changing the threshold for determining the class labels. Usually, we set $$0$$ as the threshold and whenever logit values are below 0, we say that particular input belongs to class 0 otherwise it belongs to class 1. This has the drawback that now you have to find a new hyper-parameter for deciding the threshold.
2. By oversampling from the data to make sure that each batch has same number of examples from each class. This is a good solution. However, if you have only small representations of one class, your model may not be able to generalize for that class.    

> In this blog, we will try a new approach. We will use variational auto encoder (VAE) to generate samples for the class for which we have low number of samples.
We will use these samples to train a classifier. We will compare the impact of this method and oversampling approach mentioned earlier to solve the class imbalance problem.

## Data

I used a subset of MNIST dataset to solve this problem. I created this subset by the following rules:

1. I used only two digits $$0$$ and $$6$$. I chose $$0$$ and $$6$$ because of their similarity.
2. I took $$5000$$ samples of digit $$0$$ for training
3. I took $$100$$ samples of digit $$6$$ for training.
4. This is a class imbalance problem where we have only $$2\%$$ data from one class.
5. For testing, I took images of $$0$$ and $$6$$ from MNIST test dataset.
6. The test dataset is fairly balanced. There are $$980$$ images of digit $$0$$ and $$958$$ images of digit $6$.

### Approach 1 (Class balancing by oversampling)

The approach is straight forward. To create each batch, I am taking equal samples from each class and training the neural network to minimize the cross-entropy loss for this balanced batch. The code for this code can be found [here](https://github.com/abhishm/vae_class_imbalance/blob/master/mnist_train.py)

### Approach 2 (class balancing by adding more samples from VAE)

In this approach, I first trained a VAE model on $$100$$ images of digits $$6$$. I generated $$4900$$ images from this trained VAE. I added these images to the training dataset. Then I randomly sample a batch of images from this balanced dataset to train the same model that I used in approach 1. The code for this training approach can be found [here](https://github.com/abhishm/vae_class_imbalance/blob/master/mnist_train.py).  

# Results
1. The approach 1 misclassified $$70$$ examples.
2. The approach 2 misclassified only $$30$$ examples.

# Conclusion

The VAE is able to learn the latent space from only a small set of images. This trained VAE is used to generate more training images. This extra images have sufficient variability to give a better generalized model.






























#
