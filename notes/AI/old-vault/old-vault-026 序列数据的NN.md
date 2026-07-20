---
layout: page
permalink: /notes/ai/old-vault/old-vault-026/index.html
title: 序列数据的NN
---

> Imported from old Obsidian vault on 2026-07-06. Source: `序列数据的NN.md`
tokens are the atomic indivisible units of text

GRU 门控循环单元
set gate去更新or重置 history H
重置门控制的是上一个history会有多少用在当前候选H的生成上，控制信息整合
更新门：控制新的$H_t$ 有多少权重来自$H_{t-1}$和$\tilde{H}_t$

LSTM
引入了一系列的门 遗忘门 输入门 候选记忆 输出门

