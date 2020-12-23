# MgeEditing
MgeEditing is an open source image and video editing toolbox based on MegEngine.

# 特性
* 模块化设计(models, datasets, runner, optimizer, logger, hook, evaluation), 使用注册器进行管理
* 支持多卡train、eval、test

# 说明
* 整体框架借鉴自mmediting

# model zoo

# TODO

### High
* 配置 lr_config， 通过optimizer进行手动设置

* 测试多卡

### Mid
* 完善init_weights(pretrained)的设计

* 完成test workflow  test.py -> runner.test， 进行单卡和多卡的测试

* 统计FLOPS

### Low
* optimizer支持指定到参数的设置

* 设置训练过程的seed，使得整个过程完全可复现

* add ensemble 策略 for restorers 写个装饰器？

