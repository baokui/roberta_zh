import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from bertSentEmb import *
import numpy as np
import pymysql
import os
import json
import tensorflow as tf
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
D0 = getData()
def update(path_source,bert_config_file,vocab_file,path_checcpoint):
    tf.reset_default_graph()
    with open(path_source,'r') as f:
        D1 = json.load(f)
    S = [D0[k] for k in D1]
    T = sentEmb(S,bert_config_file,vocab_file,path_checcpoint)
    keys = list(T[0][2].keys())
    R = [np.array([T[i][2][k] for i in range(len(T))]) for k in keys]
    idx = [k for k in D1]
    d0 = {idx[i]:list(R[0][i]) for i in range(len(idx))}
    d1 = {idx[i]:list(R[1][i]) for i in range(len(idx))}
    S0 = {k:[np.float(t) for t in d0[k]] for k in d0}
    S1 = {k:[np.float(t) for t in d1[k]] for k in d1}
    with open('SentVects-finetune/'+path_checcpoint.replace('/','-')+'-'+keys[0]+'.json','w') as f:
        json.dump(S0,f)
    with open('SentVects-finetune/'+path_checcpoint.replace('/','-')+'-'+keys[1]+'.json','w') as f:
        json.dump(S1,f)
def main():
    path_source = 'SentVects/roberta_zh_L-6-H-768_A-12-lastToken.json'
    modelname = 'roberta_zh_L-6-H-768_A-12'
    path_checcpoint = 'model/roberta-6-finetune/model.ckpt-141000'
    bert_config_file = 'model/{}/bert_config.json'.format(modelname)
    vocab_file = 'model/{}/vocab.txt'.format(modelname)
    update(path_source,bert_config_file,vocab_file,path_checcpoint)

    path_source = 'SentVects/roberta_zh_l12-lastToken.json'
    modelname = 'roberta_zh_l12'
    path_checcpoint = 'model/roberta-12-finetune/model.ckpt-141000'
    bert_config_file = 'model/{}/bert_config.json'.format(modelname)
    vocab_file = 'model/{}/vocab.txt'.format(modelname)
    update(path_source, bert_config_file, vocab_file, path_checcpoint)