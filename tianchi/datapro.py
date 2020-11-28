import random
path_tnews = 'data/TNEWS_train1128.csv'
path_ocnli = 'data/OCNLI_train1128.csv'
path_ocemo = 'data/OCEMOTION_train1128.csv'
if 1:
    paths = [path_tnews,path_ocnli,path_ocemo]
    C = ['*','#','$']
    D_label = {}
    M_label = {}
    path = paths[0]
    S_tn = []
    with open(path,'r',encoding='utf-8') as f:
        s = f.read().strip().split('\n')
    s = [ss.split('\t') for ss in s]
    for ss in s:
        if ss[-1] not in D_label:
            D_label[ss[-1]] = len(D_label)
            M_label[ss[-1]] = 1
        else:
            M_label[ss[-1]] += 1
    S_tn = [C[0]+ss[1]+'\t'+str(D_label[ss[-1]]) for ss in s]
    random.shuffle(S_tn)
    idx = int(len(S_tn)*0.2)
    S_tn_test = S_tn[:idx]
    S_tn = S_tn[idx:]
    ##
    path = paths[1]
    S_on = []
    with open(path,'r',encoding='utf-8') as f:
        s = f.read().strip().split('\n')
    s = [ss.split('\t') for ss in s]
    for ss in s:
        if ss[-1] not in D_label:
            D_label[ss[-1]] = len(D_label)
            M_label[ss[-1]] = 1
        else:
            M_label[ss[-1]] += 1
    def padding(s0,pad='^',maxlen=63):
        s0 = s0+(maxlen-len(s0))*pad
        s0 = s0[:maxlen]
        return s0
    S_on = [C[1]+padding(ss[1])+'&'+padding(ss[2])+'\t'+str(D_label[ss[-1]]) for ss in s]
    random.shuffle(S_on)
    idx = int(len(S_on) * 0.2)
    S_on_test = S_on[:idx]
    S_on = S_on[idx:]
    #
    path = paths[2]
    S_oe = []
    with open(path,'r',encoding='utf-8') as f:
        s = f.read().strip().split('\n')
    s = [ss.split('\t') for ss in s]
    for ss in s:
        if ss[-1] not in D_label:
            D_label[ss[-1]] = len(D_label)
            M_label[ss[-1]] = 1
        else:
            M_label[ss[-1]] += 1
    S_oe = [C[2]+ss[1]+'\t'+str(D_label[ss[-1]]) for ss in s]
    random.shuffle(S_oe)
    idx = int(len(S_oe) * 0.2)
    S_oe_test = S_oe[:idx]
    S_oe = S_oe[idx:]
if 1:
    import json
    with open('data/prepro/D_label.json','w',encoding='utf-8') as f:
        json.dump(D_label,f,ensure_ascii=False,indent=4)
    with open('data/prepro/M_label.json','w',encoding='utf-8') as f:
        json.dump(M_label,f,ensure_ascii=False,indent=4)
    S = S_oe+S_tn+S_on
    import random
    random.shuffle(S)
    with open('data/prepro/train.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(S))
    with open('data/prepro/dev_oe.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(S_oe_test))
    with open('data/prepro/dev_tn.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(S_tn_test))
    with open('data/prepro/dev_on.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(S_on_test))

if 1:
    with open('data/prepro/dev.csv', 'r', encoding='utf-8') as f:
        s = f.read().strip().split('\n')
    s = [ss.split('\t') for ss in s]
    for i in range(len(s)):
        if s[i][0][0]=='#':
            s[i][0] = s[i][0].replace('^','')
            t = s[i][0].split('&')
            t[1] = s[i][0][0]+''.join(t[1:])
            s[i] = t+[s[i][1]]
        else:
            s[i] = [s[i][0]]+[s[i][0][0]]+[s[i][1]]

with open('data/prepro/dev.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(['\t'.join(ss) for ss in s]))
