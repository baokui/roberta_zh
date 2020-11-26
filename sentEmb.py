from bertSentEmb import sentEmb
import numpy as np
from sklearn import preprocessing
import json
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
def sentEmbing(D,init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag):
    S = [d['content'] for d in D]
    T = sentEmb(S, bert_config_file, vocab_file, init_checkpoint,max_seqlen)
    R = [T[i][2]['lastToken'] for i in range(len(T))]
    V = norm(np.array(R))
    for i in range(len(D)):
        D[i][tag] = np.array(V[i]).tolist()
    return D
def main(path_data,path_target,init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag):
    D = json.load(open(path_data, 'r', encoding='utf-8'))
    D = sentEmbing(D,init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag)
    with open(path_target,'w',encoding='utf-8') as f:
        json.dump(D,f,ensure_ascii=False,indent=4)
if __name__=='__main__':
    path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag = sys.argv[2:]
    max_seqlen = int(max_seqlen)
    main(path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag)
