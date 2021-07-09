from megengine import tensor
import megengine.functional as F
import megengine as mge
import numpy as np
from megengine.autodiff import GradManager
import math

def do_attention(query, key, value, mask):
    # print(query.shape, key.shape)
    scores = F.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
    # print(scores.shape)
    eps = F.zeros_like(scores) - 1e9
    scores = F.where(mask, eps, scores)
    p_attn = F.nn.softmax(scores, axis=2)
    # print(p_attn.shape, value.shape)
    p_val = F.matmul(p_attn, value)
    return p_val, p_attn

if __name__ == "__main__":
    gm = GradManager()    
    query = mge.Parameter(np.random.random((2, 10, 10)))
    key = mge.Parameter(np.random.random((2, 10, 10)))
    value = mge.Parameter(np.random.random((2, 10, 10)))
    mask = mge.tensor(np.random.random((2, 1, 10))) > 0.5
    gm.attach([query, key, value])
    with gm:
        mask = F.broadcast_to(mask, (2, 10, 10))
        p_val, _  = do_attention(query, key, value, mask)
        loss = p_val.mean()
        gm.backward(loss)
        print(loss)
        print(query.grad)
        print(key.grad)
        print(value.grad)

