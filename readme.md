# MgeEditing
MgeEditing is an open source image and video editing toolbox based on MegEngine.

# 特性
* 模块化设计(utils, models, datasets, runner, optimizer, hook), 使用注册器进行管理，结构简单明了
* 支持多卡train、eval、test，效率高
* 丰富的小工具，包括多进程管理的logger、average pool、可视化等
* 教程丰富，适合初学者入门也适合有一定基础的同学进阶
* 采用国产框架MegEngine，可以近距离接触开发者 `doge`
# model zoo
## restorers

|  model    |  paper    |
| ---- | ---- |
| RCAN | https://arxiv.org/pdf/1807.02758.pdf |
|   RSDN   |   https://arxiv.org/pdf/2008.00455.pdf   |


## synthesizers
|  model    |  paper    |
| ---- | ---- |
|   CycleGAN   |   https://arxiv.org/abs/1703.10593   |

# TODO

### High
* 测试多卡训练，调整Logger，完善模型的load和resume

* 配置 lr_config， 通过optimizer进行手动设置

### Mid
* 完善init_weights(pretrained)的设计

* 增加统计FLOPS的代码

### Low
* 设置训练过程的seed，使得整个过程完全可复现

* 完善ensemble的设计

* 整理utils，分包

* 支持多机多卡(train)

# 说明
* 整体框架借鉴自mmediting
