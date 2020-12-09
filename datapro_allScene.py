import os
import numpy as np
path_data = 'data_allScene_pretrain/raw-washed/'
files = os.listdir(path_data)
files = [os.path.join(path_data,file) for file in files]
S = []
for i in range(len(files)):
    with open(files[i],'r',encoding='utf-8') as f:
        s = f.read().strip().split('\n\n')
    S.extend(s)

Words = {}
for s in S:
    for ss in s:
        if ss not in Words:
            Words[ss] = 1
        else:
            Words[ss] += 1
T = [[k,Words[k]] for k in Words]
T = sorted(T,key=lambda x:-x[1])

IDF = {}
N = len(S)
for t in T:
    IDF[t[0]] = np.log(N/(t[1]))
IDF['UNK'] = 10
import json
with open('data_allScene_pretrain/IDF.json','w',encoding='utf-8') as f:
    json.dump(IDF,f)