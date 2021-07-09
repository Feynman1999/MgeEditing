import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from megengine import hub

modelname = 'resnet' + str(18)
model = megengine.hub.load('megengine/models', modelname, pretrained=True)
print('=> loading pretrained model {} from megengine/models'.format(modelname))
for item in model.state_dict().keys():
    print(item)