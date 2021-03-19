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
def sentEmbing(D,init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag,IDF):
    S = [d['content'] for d in D]
    T = sentEmb(S, bert_config_file, vocab_file, init_checkpoint,max_seqlen)
    #R = [T[i][2]['lastToken'] for i in range(len(T))]
    #R = [np.mean(T[i][2]['sequence_vector'][1:len(S[i])+1],axis=0) for i in range(len(T))]
    R = []
    for i in range(len(T)):
        v0 = T[i][2]['sequence_vector']
        v = []
        for j in range(min(len(S[i]),max_seqlen-1)):
            if S[i][j] not in IDF:
                w = IDF['UNK']
            else:
                w = IDF[S[i][j]]
            v.append(w*v0[j+1])
        R.append(np.sum(np.array(v),axis=0))
    V = norm(np.array(R))
    D0 = []
    for i in range(len(D)):
        D0.append({'id':D[i]['id'],tag: np.array(V[i]).tolist()})
    return D0
def main(path_data,path_target,init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag,IDF):
    D = json.load(open(path_data, 'r', encoding='utf-8'))
    D1 = []
    i0 = 0
    while i0<len(D):
        print('process:',i0,len(D1))
        i1 = i0+10000
        D0 = sentEmbing(D[i0:i1],init_checkpoint,bert_config_file,vocab_file,max_seqlen,tag,IDF)
        D1.extend(D0)
        with open(path_target,'w',encoding='utf-8') as f:
            json.dump(D1,f,ensure_ascii=False,indent=4)
if __name__=='__main__':
    path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag, path_idf = sys.argv[2:]
    max_seqlen = int(max_seqlen)
    IDF = json.load(open(path_idf))
    main(path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag, IDF)
