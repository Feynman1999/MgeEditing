# for epoch based runner (suggested)
can resume the epoch for training, and can set epoch in the dist_sampler, so that the data order can be resumed too.

**advantage:** 
* You can reproduce the model completely, no matter how many times you save and resume it.(e.g. resume 3 times in epoch 10 20 and 40 V.S. resume 1 time in epoch 15 V.S. never stop, these results are equal) 
* Can use repeatdataset to speed up too, but repeat times can't be too large, otherwise one epoch has too mush data, even you can't save the model due to epoch is 0 always.  

**shortcoming:** It can only be saved and resumed by epoch.


# iter based runner  (have been deprecated due to Ockham's Razor)