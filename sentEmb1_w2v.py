import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from bertSentEmb import *
import numpy as np
import pymysql
import os
import json
import tensorflow as tf
import sys
sys.path.append('../../vpa/vpa-studio-research/search-online')
from SentEmb_w2v import sentemb
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
with open('SentVects/roberta_zh_L-6-H-768_A-12-lastToken.json','r') as f:
    V0 = json.load(f)

S = [D[k] for k in V0]
Ids = [k for k in V0]
V0,V1 = sentemb(S)