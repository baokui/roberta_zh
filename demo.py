import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bertSentEmb import *
def uniform(v):
    r = v/np.sqrt(v.dot(v))
    return r
key0 = 'roeberta_zh_L-24_H-1024_A-16'
bert_config_file = 'model/roberta_zh_L-6-H-768_A-12/bert_config.json'
vocab_file = 'model/roberta_zh_L-6-H-768_A-12/vocab.txt'
init_checkpoint = 'model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt'
S = ['我爱你','你爱我']
T = sentEmb(S,bert_config_file,vocab_file,init_checkpoint)

R = {S[i]:uniform(T[i][2]['lastToken']) for i in range(len(T))}
print(R[S[0]].dot(R[S[1]]))