# 简介
此模型为中国农业人工智能创新创业大赛红烧鱼七秒队伍方案(分数rank2，综合分数rank1)
比赛链接：[中国农业人工智能创新创业大赛](https://studio.brainpp.com/competition/3?name=%E4%B8%AD%E5%9B%BD%E5%86%9C%E4%B8%9A%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%88%9B%E6%96%B0%E5%88%9B%E4%B8%9A%E5%A4%A7%E8%B5%9B)

# 模型介绍
该模型基于centertrack(https://github.com/xingyizhou/CenterTrack)，并做了以下变化：
* 更改BN为group normalization
* 更改模型的stride为2 (官方为4), backbone为DLA34 with DCN
* 一些数据增强，包括color jitter、flip、RandomCropPad等，具体可以查看代码

# 数据准备
* 从官方处下载数据集，放置在任意位置即可
* 数据链接：https://data.megengine.org.cn/megstudio/MFT_Challenges.tar.gz

# 训练
## 修改配置文件
在`configs/mot/`目录下可以找到一个配置文件`centertracker_fish_dla_GN.py`，即比赛最终模型的训练配置，可以直接进行使用，也可以根据自己的情况基于其进行参数的调整
![训练配置文件图示](https://img01.sogoucdn.com/app/a/100520146/93bcb7570248c1a4a0f19a183f9bb535)
## 单机单卡启动命令
`python tools/train.py configs/mot/centertracker_fish_dla_GN.py --gpuids 0 -d`
> 0就是指使用编号为0的那个gpu, -1使用cpu
## 单机多卡启动命令
`python tools/train.py configs/mot/centertracker_fish_dla_GN.py --gpuids 4,5,6,7,8,9 -d`
> 4,5,6,7,8,9就是指使用这些编号的gpus
# 测试
## 修改配置文件
在`configs/mot/`目录下可以找到一个配置文件`centertracker_fish_dla_GN_test.py`，即比赛最终模型的测试配置，可以直接使用，也可以根据自己的情况基于其进行参数的调整
![测试配置文件图示](https://img01.sogoucdn.com/app/a/100520146/3891821abf4a2799d2c184df48b937e1)
## 单机单卡启动命令

`python tools/test.py configs/mot/centertracker_fish_dla_GN_test.py --gpuids 0 -d`
> 运行结束后可以在work_dir文件夹中找到最新的以时间戳命名的文件夹，里面存放的是结果txt

## 预训练模型

我们提供了此次比赛的结果模型：

https://pan.baidu.com/s/1kkfBGOkgejjSBQAzWvt8PQ

提取码: 4aps 



# 代码库

* 基于[MgeEditing](https://github.com/Feynman1999/MgeEditing)，欢迎大家star~

# 说明
* 此代码只在一类的情况下验证过，多类情况理论上可行但估计存在bug