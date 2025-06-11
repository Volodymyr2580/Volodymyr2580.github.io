---
layout: page
permalink: /notes/AI/RL_1/index.html
title: 强化学习的数学原理 1
---


RL系列这部分主要摘录的是我在学习《强化学习的数学原理》一书时做的笔记。原书的地址——https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning


我在思考这类笔记的对外输出价值的时候想到，其实分享有多种形式，一种是直接翻译，但这就肯定不如去看原书了；一种是你已经熟练掌握或是已经学过一遍了，然后回过头来做梳理；另一种就是像我现在想做的，就是记录我的部分学习过程，这个学习过程不光是摘录原书中的重要知识点，也能记录我在学习过程中的一些思考和想法。我觉得这也是有意义有价值的。

## 基本概念
强化学习的研究对象是agent在与环境的交互过程中如何学到最优策略。为此我们引入一些基本的概念。


第一个概念是state, 描述agent的状态，比如位置。状态的全集叫状态空间(state space)，$\mathcal{S}$


agent所能选择的行动集合由$\mathcal{A}$表示，称为行动空间(action space)，我们也可以定义一个$\mathcal{A}:\mathcal{S}\to2^\mathcal{A}$的映射，表示不同状态点上所能选择的不同的行动集合,比如$\mathcal{A}(s_1)=\{a_1,a_2\}$


当agent从一个状态移动到另一状态的过程称为state transition。我们可以用以下的符号表示agent从状态$s_1$选择行为$a_2$移动到了状态$s_2$：
$$s_1\xrightarrow{a_2}s_2$$


那我们自然关心的问题就是我们在某个状态该选择什么样的行动才能更好地完成我们的目标——这就是策略(policy)。


数学上，state transition和policy都可以用条件概率来表示，之后遇到stochastic的问题时候再回顾吧，当前考虑的policy都是deterministic的，但是学过博弈论的应该清楚很多时候我们需要混合策略才有best response。


而奖励(reward)是强化学习中最独特的一个概念，在某个state上agent选择一个action之后会收到一个reward, 记作$r(s,a)$，作为环境或任务对其行为的反馈，是state和action的函数；这更多是人为设定的用于指引我们的agent针对任务和环境选择更好的策略。


那一个很trivial的question便是，给定一个reward之后，如果让agent每一步贪心地选择最大reward执行，能否得到一个好的policy呢？这显然是不对的，因为我们不能只看当前步的reward,而是要考虑到后续的reward，or total reward。


这里或者还有一个疑问在于，比如如果有两个状态$s_1,s_2$，他们有互通的行动使得$s_1\xrightarrow{a_1}s_2, s_2\xrightarrow{a_1}s_1$，并且这两种行动都会导致正的reward,那我的agent岂不是在这两个点之间反复横跳就行？


一个轨道(trajectory)可以如下表示：

$$s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3}s_5\xrightarrow[r=0]{a_3}s_8 \xrightarrow[r=1]{a_2}s_9$$

记录了state transition和每一步的action and reward

每条轨道的回报(return){也称total rewards or cumulative rewards}定义为每一步的reward的代数求和：

$$return = 0+0+0+1=1$$

我们可以根据轨道的return来评估我们的策略是否好。return包含了我们先前说的immediate reward和future rewards，前者指从初始状态开始第一个行动立刻能收到的reward，后者指离开后得到的所有rewards。


那针对一个无穷长的轨道，如果我们单纯对reward做代数求和计算return很容易就会发散到无穷（比如我先前提起过的在两个点之间反复横跳），为此类似博弈论处理动态博弈(dynamic game)时候定义的贴现因子(discount factor)，我们通常计算一个贴现后的回报：

$$\text{discounted return} = \sum_{i=0}^{\infty}\gamma^ir_i$$

其中贴现因子$\gamma \in (0,1)$，学习过动态博弈的应该清楚这个贴现率会对子博弈完美均衡(SPE)有什么样重要的影响。直观来说，当贴现率很小的时候，我们的agent可能就不太关心后续的reward有多大，而更专注最大化眼前的收益，策略就越趋近于贪心算法。极端的，如果$\gamma=0$，那我们只需要考虑一步的最大化reward。


书上提到有个概念叫episode,表示跟随一个policy后agent可能会停在某些terminal states，最终得到的trajectory称为an episode or a trial。在我看来和轨道的定义没有什么区别。


第一章的最后提到一个Markov decision processes（MDPs）
是在先前的模型上添加了马氏链的性质，即转移概率只依赖于上一步的状态和行为，和先前的历史没关系(memoryless property)，具体的数学表示都相当trivial。


具体的元素为：状态空间$\mathcal{S}$,行动空间$\mathcal{A}$,奖励集合$\mathcal{R}(s,a)$ 
模型需要定义：（1）转移概率$p(s'|s,a)$ 表示在状态s采取行动a会转移到状态s'的概率； （2）奖励概率$p(r|s,a)$表示在状态s采取行动a会得到奖励r的概率


则一个策略policy为——$\pi(a|s)$表示在状态s选择行动a的概率，策略就是每个状态点上的$\pi$的概率分布全体。


在QA环节提到一个有意思的问题，就是上面定义的奖励只是当前state和action的函数，但我们不应该也要关心我们去到的下一个state是什么吗？那自然地会想到奖励是不是该定义为当前状态、下一状态和行动的函数呢？但其实按照上面MDPs的模型，下一状态s'会概率依赖于s,a，所以可以显式地将奖励r只表示为s和a的函数。
