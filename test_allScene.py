import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
def test(S,path_target):
    Docs = getData()
    files = os.listdir('SentVects/')
    files = [file for file in files if 'lastToken.json' in file]
    for file in files:
        D = json.load(open(os.path.join('SentVects', file), 'r'))
        print(file, len(D))
        Ids = [k for k in D]
        V = [D[k] for k in D]
        V = np.array(V)
        V = norm(V)
        idx = file.index('-lastToken')
        key0 = file[:idx]
        bert_config_file = 'model/{}/bert_config.json'.format(key0)
        vocab_file = 'model/{}/vocab.txt'.format(key0)
        init_checkpoint = 'model/{}/bert_model.ckpt'.format(key0)
        tf.reset_default_graph()
        T = sentEmb(S, bert_config_file, vocab_file, init_checkpoint)
        R = [T[i][2]['lastToken'] for i in range(len(T))]
        R = norm(np.array(R))
        score = np.dot(R, np.transpose(V))
        idx_score = np.argsort(-score, axis=-1)
        result = []
        for i in range(len(S)):
            idx = idx_score[i][:10]
            rr = [Docs[Ids[ii]] + '\t%0.4f' % score[i][ii] for ii in idx]
            result.append({'input': S[i], 'result': rr})
        with open(path_target.replace('result.txt', 'result-' + file[:-5] + '.json'), 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

path_data = 'data_allScene/20201109-all.txt'
path_target = 'data_allScene/20201109-result.txt'
with open(path_data,'r',encoding='utf-8') as f:
    S = f.read().strip().split('\n')[1:]
S = [s.split('\t')[0] for s in S[100000:200000]]
S = [s for s in S if len(s)>5 and len(s)<20]
S = S[:10000]
test(S,path_target)


S = ['自爱，沉稳，而后','部门','欢迎加入我们','十月的晚风吹','从无话不说到','怀孕时','哭的老娘的心❤都','装进我的口袋','亲爱的同学',
     '从白昼到黑夜，','医者仁心！']
path_target = 'data_allScene/test-result.txt'
test(S,path_target)
