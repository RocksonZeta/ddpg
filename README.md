# ddpg
ddpg(Deep Deterministic Policy Gradient) algorithm

## Dependency
tensorflow >= 1.4

## Introduction
Consider : ![](https://latex.codecogs.com/gif.latex?%7Bs_1%2Ca_1%2Cr_1%2C...%2Cs_T%2Ca_T%2Cr_T%7D)   
**Critic** : (state,action)  -> q value(scalar) goodness of action   
**Actor** : state -> action   

Deterministic policy :![](https://latex.codecogs.com/gif.latex?a%3D%5Cpi%28s%7C%5Ctheta%5E%7B%5Cmu%7D%29)  
Action Value function :![](https://latex.codecogs.com/gif.latex?Q%28s%2Ca%7C%5Ctheta%5EQ%29)   
Critic network is q function(action-value function),giving the goodness of actor.  Training critic network close to the fact (real reward), and supply gradient for actor update.Critic judge the actor is real(close the fact ) or fake(depart from the fact).
Policy network update action policy by replay and giving the best action. Actor always try to get the biggest reward.




## Algorithm
![](ddpg.png)
