import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bertSentEmb import *
import json
import numpy as np
with open('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/Docs-multiReplace.json','r',encoding='utf-8') as f:
    D = json.load(f)
S = [d['content'] for d in D]
T = sentEmb(S)
if len(T)!=len(S):
    print('error')
keys = list(T[0][2].keys())
R = [np.array([T[i][2][k] for i in range(len(T))]) for k in keys]
np.save('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/multiReplace'+keys[0]+'.npy',R[0])
np.save('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/multiReplace' + keys[1] + '.npy', R[1])

with open('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/Prose/contents.txt','r',encoding='utf-8') as f:
    S = f.read().strip().split('\n')
T = sentEmb(S)
if len(T)!=len(S):
    print('error')
keys = list(T[0][2].keys())
R = [np.array([T[i][2][k] for i in range(len(T))]) for k in keys]
np.save('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/Prose/prose'+keys[0]+'.npy',R[0])
np.save('/search/odin/guobk/vpa/vpa-studio-research/search/datapro/Docs/Prose/prose' + keys[1] + '.npy', R[1])
