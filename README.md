# MgeEditing
MgeEditing is an open source image and video editing toolbox based on MegEngine.

# 特性
* 模块化设计(models, datasets, runner, optimizer, logger, hook), 使用注册器进行管理
* 支持多卡train、eval、test(inference)

# 说明
整体框架借鉴自[mmediting](https://github.com/open-mmlab/mmediting)  

# model zoo
参考[configs目录](https://github.com/Feynman1999/MgeEditing/tree/master/configs)下的配置文件和md文件。

# docs
参考[docs目录](https://github.com/Feynman1999/MgeEditing/tree/master/docs)。

# TODO

### High
* 对于视频类的任务，如何对结果进行重新归类(eval)     先不考虑并行计算，先考虑有了一个list of dict，如何在dataset的evaluate方法中去统计

* 在train.py中完成test，这样就不用写重复的代码了？

* 测试多卡训练

* 配置 lr_config， 通过optimizer进行手动设置


### Mid
* 动态配置SublinearMemoryConfig

* 完善init_weights(pretrained)的设计

* 完成test workflow  test.py -> runner.test， 进行单卡和多卡的测试

* 统计FLOPS

### Low
* optimizer支持指定到参数的设置

* 设置训练过程的seed，使得整个过程完全可复现

* add ensemble 策略 for restorers 写个装饰器？

* checkpoint simlink

* 模型动转静、模型部署


