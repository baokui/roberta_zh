import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from bertSentEmb import *
import json
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import pymysql
def getData():
    conn = pymysql.connect(
        host='mt.tugele.rds.sogou',
        user='tugele_new',
        password='tUgele2017OOT',
        charset='utf8',
        port=3306,
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    # ****python, 必须有一个游标对象， 用来给数据库发送sql语句， 并执行的.
    # 2. 创建游标对象，
    cur = conn.cursor()
    # 4). **************************数据库查询*****************************
    # sqli = 'SELECT * FROM tugele.ns_flx_wisdom_words_new'
    sqli = 'SELECT a.id,a.content,a.isDeleted FROM (tugele.ns_flx_wisdom_words_new a)'
    result = cur.execute(sqli)  # 默认不返回查询结果集， 返回数据记录数。
    info = cur.fetchall()  # 3). 获取所有的查询结果
    # print(info)
    # print(len(info))
    # 4. 关闭游标
    cur.close()
    # 5. 关闭连接
    conn.close()
    S = {str(info[i][0]): info[i][1] for i in range(len(info)) if info[i][2] == 0}
    return S
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
def searching(queries,Docs,Docs0,bert_config_file, vocab_file, init_checkpoint):
    Ids = [k for k in Docs]
    V = [Docs[k] for k in Docs]
    V = np.array(V)
    V = norm(V)
    tf.reset_default_graph()
    T = sentEmb(queries, bert_config_file, vocab_file, init_checkpoint)
    R = [T[i][2]['lastToken'] for i in range(len(T))]
    R = norm(np.array(R))
    score = np.dot(R, np.transpose(V))
    idx_score = np.argsort(-score, axis=-1)
    result = []
    for i in range(len(queries)):
        idx = idx_score[i][:10]
        rr = [Docs0[Ids[ii]] + '\t%0.4f' % score[i][ii] for ii in idx]
        result.append({'input': queries[i], 'result': rr})
    return result
def main(path_query,path_doc,path_target):
    Docs0 = getData()
    path_doc = 'SentVects-finetune/model-roberta-12-finetune-model.ckpt-141000-lastToken.json'
    path_data = 'data_allScene/20201109-all.txt'
    with open(path_data, 'r', encoding='utf-8') as f:
        S = f.read().strip().split('\n')[1:]
    S = [s.split('\t')[0] for s in S[100000:200000]]
    S = [s for s in S if len(s) > 5 and len(s) < 20]
    queries = S[:10000]
    Docs = json.load(open(path_doc,'r'))
    modelname = 'roberta_zh_l12'
    path_checcpoint = 'model/roberta-12-finetune/model.ckpt-141000'
    bert_config_file = 'model/{}/bert_config.json'.format(modelname)
    vocab_file = 'model/{}/vocab.txt'.format(modelname)
    path_target = 'data_allScene/20201109-result-finetune-12layers.txt'
    result = searching(queries,Docs,Docs0,bert_config_file, vocab_file, path_checcpoint)
    with open(path_target,'w') as f:
        json.dump(result,f,ensure_ascii=False,indent=4)

def test():
    path_data = 'tianchi/data/prepro/dev.txt'
    with open(path_data, 'r', encoding='utf-8') as f:
        S = f.read().strip().split('\n')
    S = [s.split('\t') for s in S]
    init_checkpoint = 'tianchi/model/model.ckpt-3000'
    bert_config_file = 'tianchi/model/bert_config.json'
    vocab_file = 'tianchi/model/vocab.txt'
    T = sentEmb_tianchi(S, bert_config_file, vocab_file, init_checkpoint,256)
    for i in range(len(T)):
        T[i].append(np.argmax(T[i][2]))
    p = [T[i][1][-1]==str(T[i][-1]) for i in range(len(T))]


def test_pretrain64():
    path_data = 'data_allScene/20201109-all.txt'
    with open(path_data, 'r', encoding='utf-8') as f:
        S = f.read().strip().split('\n')[1:]
    S = [s.split('\t')[0] for s in S[100000:200000]]
    S = [s for s in S if len(s) > 5 and len(s) < 30]
    init_checkpoint = 'model/bert_allScene64/ckpt/model.ckpt-200000'
    bert_config_file = 'model/bert_allScene64/bert_config.json'
    vocab_file = 'model/bert_allScene64/vocab.txt'
    T = sentEmb(S, bert_config_file, vocab_file, init_checkpoint,64)
    R = [T[i][2]['lastToken'] for i in range(len(T))]
    Q = norm(np.array(R))
    Docs0 = getData()
    path_doc = 'SentVects-finetune/model-roberta-12-finetune-model.ckpt-141000-lastToken.json'
    Docs = json.load(open(path_doc, 'r'))
    D = [Docs0[k] for k in Docs0 if k in Docs]
    T = sentEmb(D, bert_config_file, vocab_file, init_checkpoint, 64)
    R = [T[i][2]['lastToken'] for i in range(len(T))]
    D_V = norm(np.array(R))
    s = Q.dot(np.transpose(D_V))
    idx_score = np.argsort(-s, axis=-1)
    result = []
    for i in range(100):
        idx = idx_score[i][:10]
        rr = [D[ii] + '\t%0.4f' % s[i][ii] for ii in idx]
        result.append({'input': S[i], 'result': rr})
    with open('data_allScene/Result-comp.json','r') as f:
        R0 = json.load(f)
    for i in range(len(result)):
        R0[i]['result-3-pretrain64'] = result[i]['result']
    with open('data_allScene/Result-comp.json','w') as f:
        json.dump(R0,f,ensure_ascii=False,indent=4)


