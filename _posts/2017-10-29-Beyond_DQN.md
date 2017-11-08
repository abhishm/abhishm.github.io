---
layout: post
title: Going beyond Deep Q-Network
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

Deep Q-Network (DQN) were first introduced four years back in the [Google Deep Mind](https://deepmind.com/) seminal [paper](https://arxiv.org/abs/1312.5602). Since its inception, there were many algorithms developed that extend DQN networks. In this blog post, we will discuss these new algorithms. We will focus mainly on three variants of DQN named as following:
1. Double DQN
2. Dueling DQN
3. Priority DQN

A modular implementation of these algorithms can be found in my [github reposiory](https://github.com/abhishm/beyond_dqn). This blog assumes that you are familiar with basics of DQN networks and Reinforcement Learning. If you are not familiar with these concept, please look at my previous [blog post](https://abhishm.github.io/DQN/) on these topics.  

# [Double DQN](https://papers.nips.cc/paper/3964-double-q-learning.pdf)

In reinforcement learning problems, the optimal action values satisfy the Bellman equations described as following:

$$
Q^*(s, a) = r + \gamma \max_{b} E_{s'}[Q^*(s', b)].
$$

Q-learning is inspired by approximating Bellman equation. In Q-learning, we start with arbitrary $$Q-$$values for each state-action pair. We collect a trajectory and we change these Q-values according to the following equation:

$$
Q(s, a) \leftarrow (1 - \alpha)Q(s, a) + \alpha (r + \gamma \max_b Q(s', b))
$$

where $$\alpha$$ is the learning rate, $$r$$ is the immediate reward after taking action $$a$$ at sate $$s$$, $$\gamma$$ is the discount factor, and $$s'$$ is the next state seen in the trajectory. Note that if we make sufficient updates to $$Q-$$values according to the above equation, the $$Q-$$values will converge to an estimate that satisfies the following equation

$$
Q'(s, a) = r + \gamma  E_{s'}[\max_{b} Q'(s', b)]
$$


Note that in the above estimate, expectation and max operator has switched places compare to their position in the Bellman equation. However, if you do sufficient exploration and updates, the action-values estimate learned by Q-learning converges towards the optimal action-values estimate $$Q^*$$. But in practice, we have limited exploration times, so the action-values learned by Q-learning are usually an overestimate of the true action-values of the optimal policy. This overestimation of action-values can caused the Q-learning to find a sub-optimal policy.

To overcome this problem, the double Q-learning algorithm was proposed in this [paper](https://papers.nips.cc/paper/3964-double-q-learning.pdf). In essence, the double Q-learning algorithm says that learn two Q-values ($$\hat{Q}(s, a)$$ and $$\tilde{Q}(s, a)$$) alternatively with slight modification of the Q-learning update rule noted as below.

$$
\hat{Q}(s, a) \leftarrow (1 - \alpha)\hat{Q}(s, a) + \alpha (r + \gamma \hat{Q}(s', \arg\!\max_b \tilde{Q}(s', b))
$$

$$
\tilde{Q}(s, a) \leftarrow (1 - \alpha)\tilde{Q}(s, a) + \alpha (r + \gamma \tilde{Q}(s', \arg\!\max_b \hat{Q}(s', b))
$$

Mainly, to update $$\tilde{Q}$$, use the optimal actions according to $$\hat{Q}(s, a)$$ and vice-versa. The [paper](https://papers.nips.cc/paper/3964-double-q-learning.pdf) suggests a periodic update where you alternatively switch between updating $$\hat{Q}$$ and $$\tilde{Q}$$.
The authors proves that by using the two estimates of Q-learning, you can overcome the problem of switching between max and expectation operator and the new method provides an unbiased estimate of $$\max_b E_{s'}[Q^*(s', b)]$$.

Note that in DQN network, the update rule is as following:

$$
w = \arg\!\min\left(f^w(s, a) - \left(r + \max_b(f^{w’}(s’, b)\right) \right)^2
$$

where $$w$$ and $$w'$$ represent the weights of Q-network and target-network respectively and $$f$$ is the neural-network architecture that is used as function approximator for Q-values.

The Double DQN approach  is just a slight modification of the above equation where we will use the Q-network to find the actions for the update as shown in the following equation:

$$
w = \arg\!\min\left(f^w(s, a) - \left(r + f^{w’}\left(s’, \arg\!\max_bf^w(s, b) \right)\right) \right)^2
$$

We can incorporate this change with a slight modification of the code:

```python
if self.use_double_dqn:
    actions_for_target = tf.argmax(self.q_values, axis=1)
    zero_axis = tf.range(tf.shape(self.q_values, out_type=tf.int64)[0])
    max_indices = tf.stack((zero_axis, actions_for_target), axis=1)
    self.max_target_q_values = tf.gather_nd(self.target_q_values, max_indices)
else:
  self.max_target_q_values = tf.reduce_max(self.target_q_values, axis=1)
```

### Results
I used to train an agent to play pong game using the Double-Q learning algorithm.
According to openai gym documentation, the pong problem is defined as following:

>  In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3). Each action is repeatedly performed for a duration of $$k$$ frames, where $$k$$ is uniformly sampled from $$\{2,3,4\}$$.

![double_q_network]({{site.baseurl}}/assets/images/2017-10-29-beyond_dqn/double_dqn_reward.png)

As you can from the above graph, the network was able to consistently win the game after 2 days of training. I haven't played with exploration parameter of $$\epsilon-$$greedy policy but I think it can be used to improve the performance of the algorithm rapidly especially by introducing a decay.

# Dueling network

Dueling network were first introduced in the [paper](https://arxiv.org/abs/1511.06581). The core idea in the Dueling networks lies in the following equation:

$$
Q(s, a) = A(s, a) + V(s)
$$

The above equation says that Q-values can be written as sum of advantage values $$A(s, a)$$ and value function $$V(s)$$. As it can be seen from the equation, the advantage function tells us the `goodness` of an action at a state compare to the average total reward at this state $$V(s)$$.

The above paper tells us that instead of computing the Q-values directly, we will compute the advantage $$A(s, a)$$ and values $$V(s)$$ first and then combine them to get the Q-values. Other than, this architectural change, the rest of the learning process is same as in the Q-learning. This architectural change in the network can be seen in the following figure.

 ![dueling_network_architecture]({{site.baseurl}}/assets/images/2017-10-29-beyond_dqn/dueling_network_architecture.png)

 The benefit of doing this architectural change is explained in the following figure in the paper.

 ![dueling_network_benefit]({{site.baseurl}}/assets/images/2017-10-29-beyond_dqn/dueling_network_benefit.png)

The above figure is the saliency map of a trained model. These maps tell us that the most important pixels that are responsible for high activations in value-function and advantage-function. The left two figures are corresponding to a state where there is no immediate danger to the player and all the actions are safe. In this case, value function is looking at far in the future and focusing on a car that can be a potential threat in future while the saliency map of advantage-function does not show any importance to any pixel. In the right two figures, there are immediate dangers to the agent and the advantage-function learns to focus on those pixels that can cause these immediate danger while value-function is still focusing on the future rewards.

 In essence, the above figure tells us that the value network learns to look in far future and the advantage function learns to find the optimal action for the immediate future. By distributing the learning of immediate and future rewards to advantage and value-function respectively, dueling network is known to learn faster. It is also evident in our experiment as shown in the figure below.   

![dueling_dqn_reward]({{site.baseurl}}/assets/images/2017-10-29-beyond_dqn/dueling_dqn_reward.png)

As it can be seen from the above figure, dueling agent learns to play pong effectively within two hours of training.
