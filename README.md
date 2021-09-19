# Decision Making Under Uncertainty with POMDPs.jl

<!-- julia blue 4063D8
julia green 389826
julia purple 9558B2
julia red CB3C33 -->

[![Julia Academy](https://img.shields.io/badge/julia%20academy-POMDPs.jl-4063D8)](https://juliaacademy.com/courses/decision-making-under-uncertainty-with-pomdps-jl)


Introduction to the [`POMDPs.jl`](https://github.com/JuliaPOMDP/POMDPs.jl) framework and its ecosystem.

<kbd>
<p align="center">
  <a href="https://github.com/mossr/julia-tufte-beamer/blob/julia-academy/pomdps.jl/main.pdf">
    <img src="./media/cover.svg"/>
  </a>
</p>
</kbd>

<br/>
<br/>

**Topics include:**
- [_Decision making under uncertainty_](youtube.com)
- [_Markov decision processes (MDPs)_](youtube.com)
- [_Partially observable Markov decision processes (POMDPs)_](youtube.com)
- [_State estimation_](youtube.com)
  - Particle filtering
- [_Reinforcement learning_](youtube.com)
  - Q-learning, SARSA
  - Value function approximation
- [_Deep reinforcement learning_](youtube.com)
  - Proximal policy optimization (PPO), deep Q-networks (DQN)
- [_Imitation learning_](youtube.com)
  - Behavior cloning
- [_Black-box validation_](youtube.com)
  - Adaptive stress testing

# Lectures

## 0. Introduction

[![Julia Academy](https://img.shields.io/badge/introduction-lecture-CB3C33)](youtube.com) <!-- TODO -->
[![Slides](https://img.shields.io/badge/introduction-slides-9558B2)](https://github.com/mossr/julia-tufte-beamer/blob/julia-academy/pomdps.jl/main.pdf)

Brief introduction to the content of this course.

<kbd>
<p align="center">
  <a href="https://github.com/mossr/julia-tufte-beamer/blob/julia-academy/pomdps.jl/main.pdf">
    <img src="./media/problems-slide.svg"/>
  </a>
</p>
</kbd>


## 1. MDPs: Markov Decision Processes

[![Julia Academy](https://img.shields.io/badge/MDPs-lecture-CB3C33)]() <!-- TODO -->
[![Slides](https://img.shields.io/badge/MDPs-slides-9558B2)](https://github.com/mossr/julia-tufte-beamer/blob/julia-academy/pomdps.jl/main.pdf)
[![Pluto](https://img.shields.io/badge/MDPs-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/1-MDPs.jl.html)

Introduction to MDPs using the _Grid World_ problem.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/1-MDPs.jl.html">
    <img src="./notebooks/gifs/gridworld_vi_γ.gif"/>
    <br/>
    <br/>
    <br/>
    <img src="./media/gridworld-transition.svg"/>
  </a>
</p>


## 2. POMDPs: Partially Observable Markov Decision Processes

[![Julia Academy](https://img.shields.io/badge/POMDPs-lecture-CB3C33)]() <!-- TODO -->
[![Slides](https://img.shields.io/badge/POMDPs-slides-9558B2)](https://github.com/mossr/julia-tufte-beamer/blob/julia-academy/pomdps.jl/main.pdf) <!-- TODO -->
[![Pluto](https://img.shields.io/badge/POMDPs-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/2-POMDPs.jl.html)

Introduction to POMDPs using the _Crying Baby_ problem.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/2-POMDPs.jl.html">
    <img src="./media/crying-baby.svg"/>
    <img src="./media/alpha-vectors.svg"/>
  </a>
</p>


## 3. State Estimation using Particle Filtering

[![Julia Academy](https://img.shields.io/badge/state%20estimation-lecture-CB3C33)]() <!-- TODO -->
[![Pluto](https://img.shields.io/badge/state%20estimation-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/3-ParticleFilters.jl.html)

Using beliefs to estimate the state of an agent through _particle filtering_.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/3-ParticleFilters.jl.html">
    <img src="./notebooks/gifs/particle_filter.gif"/>
  </a>
</p>


## 4. Approximate Methods

[![Julia Academy](https://img.shields.io/badge/approximate%20methods-lecture-CB3C33)]() <!-- TODO -->
[![Pluto](https://img.shields.io/badge/approximate%20methods-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/4-Approximate-Methods.jl.html)

Approximating a continuous space using grid interpolation and value function approximation.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/4-Approximate-Methods.jl.html">
    <img src="./notebooks/gifs/mountaincar.gif"/>
    <img src="./media/discretized-grid.svg"/>
    <img src="./media/mountaincar-value-policy.svg"/>
  </a>
</p>


## 5. Deep Reinforcement Learning

[![Julia Academy](https://img.shields.io/badge/deep%20RL-lecture-CB3C33)]() <!-- TODO -->
[![Pluto](https://img.shields.io/badge/deep%20RL-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/5-Deep-Reinforcement-Learning.jl.html)

Introduction to _deep reinforcement learning_ applied to the pendulum swing-up MDP.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/5-Deep-Reinforcement-Learning.jl.html">
    <img src="./notebooks/gifs/pendulum_ppo.gif"/>
    <img src="./media/deep-rl-curves.svg"/>
  </a>
</p>


## 6. Imitation Learning

[![Julia Academy](https://img.shields.io/badge/imitation%20learning-lecture-CB3C33)]() <!-- TODO -->
[![Pluto](https://img.shields.io/badge/imitation%20learning-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/6-Imitation-Learning.jl.html)

Introduction to _imitation learning_ using _behavior cloning_ of expert demonstrations.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/6-Imitation-Learning.jl.html">
    <img src="./notebooks/gifs/pendulum-behavior-cloned.gif"/>
    <img src="./media/behavior-cloned-curves.svg"/>
  </a>
</p>


## 7. Black-Box Validation

[![Julia Academy](https://img.shields.io/badge/black&dash;box%20validation-lecture-CB3C33)]() <!-- TODO -->
[![Pluto](https://img.shields.io/badge/black&dash;box%20validation-notebook-389826)](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/7-BlackBox-Validation.jl.html)

Stress testing a black-box system using _adaptive stress testing_.

<p align="center">
  <a href="https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/7-BlackBox-Validation.jl.html">
    <img src="./media/ast-reward.png"/>
    <img src="./media/ast.png"/>
  </a>
</p>


---
Created and taught by [Robert Moss](https://github.com/mossr).
