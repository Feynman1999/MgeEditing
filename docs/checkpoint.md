# for iter based runner
can resume the iter for training, but can not set epoch in the dist_sampler. 
why do not set? there are two reasons:
* we do not maintain the epoch when using iter based runner.Because even if maintain the epochs,
due to use repeatdataset(e.g. repeat 1000times), the epoch will change slowly.
* maintain the epoch make iterrunner more complex and more like the epochrunner, it is not necessary.

**advantage:** use the repeatdataset can speed up dataload time, especially when the data loading time is long but the dataset is small.

**shortcoming:** we do not save or maintain the epoch value.So when resume, the data input order is Unchanging, due to the epoch in the distsampler start from 0 always.But the effect of accuracy can generally be ignored. (It just can't be reproduced completely unless you know where the resume was done before and you reset the sample.epoch to zero by hand)

# for epoch based runner (suggested)
can resume the epoch for training, and can set epoch in the dist_sampler, so that the data order can be resumed too.

**advantage:** 
* You can reproduce the model completely, no matter how many times you save and resume it.(e.g. resume 3 times in epoch 10 20 and 40 V.S. resume 1 time in epoch 15 V.S. never stop, these results are equal) 
* Can use repeatdataset to speed up too, but repeat times can't be too large, otherwise one epoch has too mush data, even you can't save the model due to epoch is 0 always.  

**shortcoming:** It can only be saved and resumed by epoch.A little bit more computation for config hooks (checkpoint by epoch、lr config by epoch、log config by iter、eval config by iter)


