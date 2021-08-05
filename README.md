VisionBasedAIforShooterLand
==

## Introduction
本项目使用非侵入式的方法对安卓移动设备进行控制和交互，实现自动化的《射手大陆》游戏AI。在本项目中，我们使用固定规则，模仿学习，强化学习三种算法进行控制。

## Environment Configuration

```
conda create -n env python=3.7
conda activate env
pip install -r requirements.txt
```

## File Structure

- 固定规则 PathSearch
- 模仿学习 BehaviorCloning
- 强化学习 RL
  - 数据收集端 client
  - 模型训练端 server

## demo

![image](/docs/demo.gif)
