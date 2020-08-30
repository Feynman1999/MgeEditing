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
* 完成many to many 的训练， 考虑怎么减少io时间。

* test hook 完成test，这样就不用写重复的代码了？

* 完成视频的ensemble策略(eval and test)

* 配置 lr 调度

* 测试多卡训练

### Mid
* 动态配置SublinearMemoryConfig

* 完善init_weights(pretrained)的设计

* 统计FLOPS

### Low
* optimizer支持指定到参数的设置

* 设置训练过程的seed，使得整个过程完全可复现

* checkpoint simlink

* 模型动转静、模型部署


aistudio操作流程：按指定路径去activate mge环境，然后即可运行。


第16 23 52 59 62 72 88帧 LR多1 ，但使用windows上的ffmpeg时帧数正常。
处理流程：
1. 将HR的mp4/mkv转为LR，指令：ffmpeg -i 16.mkv -vf scale=iw/4:ih/4 -c:v libx264 -preset slow -crf 21 test.mp4
2. 将HR和LR的视频分别转为pngs，指令： ffmpeg -i test.mp4  %d.png   (1 index, name no padding)