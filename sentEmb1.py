import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bertSentEmb import *
import json
import numpy as np
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
D = getData()
key0 = 'roberta_zh_L-6-H-768_A-12'
bert_config_file = 'model/roberta_zh_L-6-H-768_A-12/bert_config.json'
vocab_file = 'model/roberta_zh_L-6-H-768_A-12/vocab.txt'
init_checkpoint = 'model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt'
idx = [k for k in D]
S = [D[k] for k in D]
T = sentEmb(S,bert_config_file,vocab_file,init_checkpoint)
if len(T)!=len(S):
    print('error')
keys = list(T[0][2].keys())
R = [np.array([T[i][2][k] for i in range(len(T))]) for k in keys]
D0 = {idx[i]:list(R[0][i]) for i in range(len(idx))}
D1 = {idx[i]:list(R[1][i]) for i in range(len(idx))}
S0 = {k:[np.float(t) for t in D0[k]] for k in D0}
S1 = {k:[np.float(t) for t in D1[k]] for k in D1}
with open('SentVects/'+key0+'-'+keys[0]+'.json','w',encoding='utf-8') as f:
    json.dump(S0,f)
with open('SentVects/'+key0+'-'+keys[1]+'.json','w') as f:
    json.dump(S1,f)
