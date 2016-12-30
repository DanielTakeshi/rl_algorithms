# Reinforcement Learning Algorithms

I will use this repository to implement various reinforcement learning algorithms, because I'm sick of reading them but not really *getting* them. Hence, hopefully this repository will help me understand them better. I will also implement various supporting code as needed, such as for simple custom scenarios like GridWorld. Or I can use OpenAI gym.

Here are the algorithms which I already know quite well and do not plan to implement:

- Value Iteration and Policy Iteration
- Q-Learning, tabular version 
- SARSA, tabular version
- Q-Learning with linear function approximation (implemented in closed-source code, sorry)
- Deep Q-Networks, vanilla version (understood via [spragnur's deep_q_rl code](https://github.com/spragunr/deep_q_rl) as well as [my personal fork](https://github.com/DanielTakeshi/deep_q_rl))

Here are the algorithms currently implemented or in progress (indicated with WIP):

- Learning with Options (WIP)
- G-Learning (WIP)

# References

I have read a number of paper references to help me out. Here are the papers that I have read, numbered on a (1) to (5) scale where a (1)  means I essentially haven't read it yet, while a (5) means I feel confident that I understand almost everything about the paper. Within a single year, these papers should be organized according to publication date, which gives an idea of how these contributions were organized. Summaries for some of these papers are [in my paper notes repository](https://github.com/DanielTakeshi/Paper_Notes).

### 2017

- [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning (under review)](https://arxiv.org/abs/1611.04717) (1)
- [RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning (under review)](https://arxiv.org/abs/1611.02779) (4) [[Summary](https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/RL2-Fast_Reinforcement_Learning_via_Slow_Reinforcement_Learning.md)]
- Composing Meta-Policies for Autonomous Driving Using Hierarchical Deep Reinforcement Learning (under review) (5)
- Multilateral Surgical Pattern Cutting in 2D Orthotropic Gauze with Deep Reinforcement Learning Policies for Tensioning (under review) (5)

### 2016

- [Value Iteration Networks, NIPS 2016](https://arxiv.org/abs/1602.02867) (1)
- [Deep Exploration via Bootstrapped DQN, NIPS 2016](https://arxiv.org/abs/1602.04621) (1)
- [VIME: Variational Information Maximizing Exploration, NIPS 2016](https://arxiv.org/abs/1605.09674) (1)
- [Cooperative Inverse Reinforcement Learning, NIPS 2016](https://arxiv.org/abs/1606.03137) (1)
- [Principled Option Learning in Markov Decision Processes, EWRL, 2016](https://arxiv.org/abs/1609.05524) (4) [[Summary](https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Principled_Option_Learning_in_Markov_Decision_Processes.md)]
- [Taming the Noise in Reinforcement Learning via Soft Updates, UAI 2016](https://arxiv.org/abs/1512.08562) (4) [[Summary](https://github.com/DanielTakeshi/Paper_Notes/blob/master/reinforcement_learning/Taming_the_Noise_in_Reinforcement_Learning_via_Soft_Updates.md)]
- [Benchmarking Deep Reinforcement Learning for Continuous Control, ICML 2016](https://arxiv.org/abs/1604.06778) (3)
- [Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016](https://arxiv.org/abs/1511.06581) (1)
- [Asynchronous Methods for Deep Reinforcement Learning, ICML 2016](https://arxiv.org/abs/1602.01783) (3)
- [Prioritized Experience Replay, ICLR 2016](https://arxiv.org/abs/1511.05952) (1)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation, ICLR 2016](https://arxiv.org/abs/1506.02438) (1)
- [Continuous Control with Deep Reinforcement Learning, ICLR 2016](https://arxiv.org/abs/1509.02971) (1)
- [HIRL: Hierarchical Inverse Reinforcement Learning for Long-Horizon Tasks with Delayed Rewards, arXiv 2016](https://arxiv.org/abs/1604.06508) (4)
- [End-to-End Training of Deep Visuomotor Policies, JMLR 2016](https://arxiv.org/abs/1504.00702) (1)
- [Deep Reinforcement Learning with Double Q-learning, AAAI 2016](https://arxiv.org/abs/1509.06461) (2)
- [Mastering the Game of Go with Deep Neural Networks and Tree Search, Nature 2016](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) (1)

### 2015 and Earlier

- [Trust Region Policy Optimization, ICML 2015](https://arxiv.org/abs/1502.05477) (2)
- [Human-Level Control Through Deep Reinforcement Learning, Nature 2015](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) (5)
- [Playing Atari with Deep Reinforcement Learning, NIPS Workshop 2013](https://arxiv.org/abs/1312.5602) (5)
- [A Tutorial on Linear Function Approximators for Dynamic Programming and Reinforcement Learning, Foundations and Trends in Machine Learning 2013](http://www.research.rutgers.edu/~thomaswa/pub/Geramifard13Tutorial.pdf) (4)
