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
* 完成视频的ensemble策略(for test now)

* 考虑怎么减少many to many的io时间（ 部分存在内存中  \  制作lmdb）

* 检查初始化是否合理

* many to many eval时不能用多个clip，否则报错；若去掉trace则显存不足。方案：先使用一个clip eval （搞明白原因先，是因为代码bug还是shape不对应）  显存不足（显存增长的bug)： 中间结果用numpy存储试一下
对于test将10个clip分成3组，单独静态图测试（或者使用动态图，但有显存一直增加的bug）

* 配置 lr 调度

* 测试多卡训练

### Mid
* 动态配置SublinearMemoryConfig

* 完善init_weights(pretrained)的设计

* 统计FLOPS

### Low
* 视频的ensemble策略(for eval)

* optimizer支持指定到参数的设置

* 设置训练过程的seed，使得整个过程完全可复现

* checkpoint simlink

* 模型动转静、模型部署


* aistudio操作流程：按指定路径去activate mge环境，然后即可运行。
source activate /home/aistudio/external-libraries/envs/mge
缺点：shm太小
有点：有数据集，显卡32GB，性能好

* mgestudio
缺点：性能一般
优点：shm大，可以正常使用

* aimax
缺点：性能一般
优点：shm大，可以正常使用

* 级链
缺点：贵
优点：内存大


第16 23 52 59 62 72 88帧 LR多1 ，但使用windows上的ffmpeg时帧数正常。
处理流程：
1. 将HR的mp4/mkv转为LR，指令：ffmpeg -i 16.mkv -vf scale=iw/4:ih/4 -c:v libx264 -preset slow -crf 21 test.mp4
2. 将HR和LR的视频分别转为pngs，指令： ffmpeg -i test.mp4  %d.png   (1 index, name no padding)
还不清楚和官方是否一样