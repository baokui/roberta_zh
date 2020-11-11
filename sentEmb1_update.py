import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
D = getData()
files = os.listdir('SentVects/')
for file in files:
    if 'lastTokenDense' not in file:
        continue
    idx = file.index('-lastTokenDense')
    key0 = file[:idx]
    D0 = json.load(open(os.path.join('SentVects', key0+'-lastTokenDense.json'), 'r'))
    D1 = json.load(open(os.path.join('SentVects', key0 + '-lastToken.json'), 'r'))
    bert_config_file = 'model/{}/bert_config.json'.format(key0)
    vocab_file = 'model/{}/vocab.txt'.format(key0)
    init_checkpoint = 'model/{}/bert_model.ckpt'.format(key0)
    idx = [k for k in D if k not in D0]
    S = [D[k] for k in D if k not in D0]
    tf.reset_default_graph()
    T = sentEmb(S,bert_config_file,vocab_file,init_checkpoint)
    keys = list(T[0][2].keys())
    R = [np.array([T[i][2][k] for i in range(len(T))]) for k in keys]
    d0 = {idx[i]:list(R[0][i]) for i in range(len(idx))}
    d1 = {idx[i]:list(R[1][i]) for i in range(len(idx))}
    S0 = {k:[np.float(t) for t in d0[k]] for k in d0}
    S1 = {k:[np.float(t) for t in d1[k]] for k in d1}
    n0 = [len(D0),len(D1)]
    D0.update(S0)
    D1.update(S0)
    n1 = [len(D0),len(D1)]
    print('update for %s'%file)
    print('before update:%d,%d'%(n0[0],n0[1]))
    print('after update:%d,%d' % (n1[0], n1[1]))
    with open('SentVects/'+key0+'-'+keys[0]+'.json','w',encoding='utf-8') as f:
        json.dump(D0,f)
    with open('SentVects/'+key0+'-'+keys[1]+'.json','w') as f:
        json.dump(D1,f)
