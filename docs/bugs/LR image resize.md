## opencv resize VS Pillow resize
在**commit 88e57667c7c54**之前，我们使用的是opencv bicubic方法(both DIV2K and Set5)，使用的模型是DBPN，但是结果却要低于之前
在torch上做的结果两个多db。后来发现原来是resize生成LR的方式不同。

我们以之前在torch上train的模型为例，其采用Pillow的bicubic方法生成DIV2k和set5的LR，并在DIV2k上train，在set5上eval：

**epoch 50 results:**

PSNR: 25.805175625405496(将torch训练的模型在opencv生成的LR上进行测试)

PSNR: 29.706839587509457(将torch训练的模型在PIL生成的LR上进行测试)

可见eval集的分布和训练集的分布明显有gap。

## 但为啥都用opencv去训练和测试，结果会低两个多db呢？
因为opencv生成的LR保留的信息较pillow少，因此PSNR上不去。