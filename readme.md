# MgeEditing
MgeEditing is an open source image and video editing toolbox based on MegEngine.

# 特性
* 模块化设计(utils, models, datasets, runner, optimizer, hook), 使用注册器进行管理，结构简单明了
* 支持多卡train、test，效率高（多卡test需要用户清楚dataloader的过程，适情况而定）
* 丰富的小工具，包括多进程管理的logger、average pool、可视化等
* 教程丰富，适合初学者入门也适合有一定基础的同学进阶
* 采用国产框架MegEngine，可以近距离接触开发者 `doge`

# model zoo
## restorers

|  model    |  paper    | appear |
| ---- | ---- |  ----|
| basicVSR | https://arxiv.org/abs/2012.02181 | CVPR2021

## synthesizers
|  model    |  paper    | appear | illustration 
| ---- | ---- | ----| ---- |
|   STTN   |   https://arxiv.org/abs/2007.10247   | ECCV2020| still refining

## MOT(multi objects tracking)
|  model    |  paper    | appear | illustration 
| ---- | ---- | ----| ---- |
|   centertrack   |   http://arxiv.org/abs/2004.01177   | ECCV2020 | still refining


# TODO

### High
* 设置训练过程的seed，使得整个过程完全可复现
* 优化train/test的流程,在test时不需要optimizer的设置

### Mid
* 增加统计FLOPS的代码

### Low
* 整理utils，分包
* 支持多机多卡(train)

# 说明
* 整体框架借鉴自mmediting
