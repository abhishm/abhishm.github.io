---
layout: post
title: Vanila Policy Gradient with a Recurrent Neural Network Policy
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

# Multi-arm Bandit problem

Multi-arm bandit is a colorful name for a problem we daily face in our lives given choices. The problem is how to choose given multitude of options. Lets make the problem concrete. Assume that its Friday evening and you are planning to go to a fancy restaurant. Should you try your favorite restaurant or try some new restaurant. If you go for your favorite restaurant then you are `exploiting` your past knowledge and if you go for a new  restaurant, then you are `exploring` which may be useful in future to find a better restaurant than your present restaurant. However, the main question is how to balance the tradeoff between exploration and exploitation. `Thompson Sampling` essentially provide a way to solve this problem.

# Thompson Sampling

Assume that there are two restaurants $$A$$ and $$B$$. Assume that restaurant $$A$$ on an average $$\theta_1$$ times align to your taste and restaurant $$B$$ on an average $$\theta_2$$ times align to your taste. In other words, if you go to restaurant $$A$$ $$N$$ times then you will like the experience $$\theta_1 N$$ times and if you go to restaurants $$B$$ $$N$$ times then you will like the experience $$\theta_2 N$$ times where $$\theta_1$$ and $$\theta_2$$ are some numbers between $$0$$ and $$1$$. Unfortunately these numbers $$\theta_1$$ and $$\theta_2$$ are not known to you so you cannot know which restaurant you should choose. You go to restaurant very often and you want to maximize your chances of having good experiences in the restaurants during your life time. How should you choose these restaurants based on your experiences so far? Thompson Sampling comes to rescue here.


The idea behind Thompson Sampling is inspired by Bayesian Inference. Lets try to present the main idea behind Thompson Sampling as succinctly as possible below:


1. Lets assume that we have priors on unknown parameters that affects the reward for our bandit problem. In our restaurant example, the parameters are $$\theta_1$$ and $$\theta_2$$. The reward for bandit problem is essentially $$\theta_1$$ and $$\theta_2$$ which are the measure of goodness of the restaurant.
2. We sample a value of unknown parameters from this prior distribution.
3. We compute the reward from these sampled parameters.
4. We choose the actions which gives the highest reward.
5. We observe the actual reward gathered by taking our action.
6. We update the priors on the parameters using the observed reward.
7. We repeat the above procedure using the new posterior distribution.

# Thompson Sampling in Action

Now lets apply the Thompson Sampling to our restaurant hunting problem and see how does it help us in resolving the problem.
1. Since we don't know anything about the restaurants initially, we can assume that the $$\theta_1$$ and $$\theta_2$$ parameters are uniformly distributed between $$0$$ and $$1$$. Note that we can also write a uniform distribution between $$0$$ and $$1$$ as a beta-distribution $$B(1, 1)$$. Please read more about the beta-distribution [here](https://en.wikipedia.org/wiki/Beta_distribution) which is going to be helpful in understanding the upcoming text.

2. We sample a value for $$\theta_1$$ and a value for $$\theta_2$$ from their distribution. We choose the restaurant having the higher $$\theta$$ values.

3. Assume that you chose the restaurant $$A$$ in the previous step. You go to the restaurant and if you have the good experience in the restaurant, you give it $$+1$$ reward otherwise you would give it a $$0$$ reward.

4. Now we need to update the distribution for $$\theta_1$$ based on the reward that we observed in the previous step. Since our prior distribution was the beta-distribution, it is easy to do as described below:
> If your prior distribution for $$\theta_1$$ is $$B(\alpha, \beta)$$ you receive the reward $$r$$ for going to restaurant $$A$$ then the posterior probability distribution for $$\theta_1$$ is $$B(\alpha + r, \beta + (1 -r))$$.

5. Now to choose a restaurant second time, you sample new values for $$\theta_1$$ and $$\theta_2$$ but you use the updated probability distribution for the restaurant $$A$$ that was chosen in the first trail and you keep going on like that to choose restaurants.
