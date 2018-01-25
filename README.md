# ddpg
ddpg(Deep Deterministic Policy Gradient) algorithm

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Dependency
tensorflow >= 1.4

## Introduction
Consider : \\({s_1,a_1,r_1,...,s_T,a_T,r_T}\\)   
**Critic** : (state,action)  -> q value(scalar) goodness of action   
**Actor** : state -> action   

Deterministic policy :\\(a=\pi(s|\theta^{\mu})\\)   
Action Value function :\\(Q(s,a|\theta^Q)\\)   
Critic network is q function(action-value function),giving the goodness of actor.  Training critic network close to the fact (real reward), and supply gradient for actor update.Critic judge the actor is real(close the fact ) or fake(depart from the fact).
$$L=E[(R-Q(S,A|\theta^Q))^2] = \frac{1}{N}\sum_i(r_i-Q(s_i,a_i|\theta^Q)^2)$$
Policy network update action policy by replay and giving the best action. Actor always try to get the biggest reward.


## Algorithm
![](ddpg.png)
