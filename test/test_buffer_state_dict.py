import megengine

path = "./workdirs/sttn_official_fvi/20210521_164246/checkpoints/epoch_8/discriminator_module.mge"

res = megengine.load(path)

print(res.keys())

