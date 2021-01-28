from bertSentEmb import *
import numpy as np
from sklearn import preprocessing
from flask import Flask, request, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey
import json
import logging
import sys
monkey.patch_all()
app = Flask(__name__)
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
path_config = sys.argv[1]
port = int(sys.argv[2])
config = json.load(open(path_config,'r'))
bert_config_file = config['bert_config_file']
vocab_file = config['vocab_file']
max_seq_length=config['max_seq_length']
init_checkpoint=config['init_checkpoint']
path_idf=config['path_idf']
IDF = json.load(open(path_idf))
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)
label_list = ['0','1']
tf.reset_default_graph()
input_ids = tf.placeholder(tf.int32,shape = [None,max_seq_length],name = 'input_ids')
input_mask = tf.placeholder(tf.int32,shape = [None,max_seq_length],name = 'input_mask')
segment_ids = tf.placeholder(tf.int32,shape = [None,max_seq_length],name = 'segment_ids')
labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
sequence_output,output_layer0,loss, per_example_loss, logits, probabilities = create_model(bert_config, False, input_ids, input_mask, segment_ids,labels, 2, False)
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names
 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
if len(assignment_map)==0:
    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
init_vars = tf.train.list_variables(init_checkpoint)
print("tarvs",tvars)
print("init_vars",init_vars)
print("assignment_map",assignment_map)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
@app.route('/api/bertEmb', methods=['POST'])
def test1():
    r = request.json
    inputStr = r["input"]
    result = []
    try:
        text_a = inputStr
        example = InputExample(guid='guid', text_a=text_a, label='0')
        feature = convert_single_example(10, example, label_list, max_seq_length, tokenizer)
        feed_dict = {input_ids: [feature.input_ids], segment_ids: [feature.segment_ids],
                     input_mask: [feature.input_mask]}
        v0 = sess.run(sequence_output, feed_dict=feed_dict)[0]
        v = []
        for j in range(min(len(inputStr), max_seq_length - 1)):
            if inputStr[j] not in IDF:
                w = IDF['UNK']
            else:
                w = IDF[inputStr[j]]
            v.append(w * v0[j + 1])
        v = np.sum(np.array(v), axis=0)
        v = v / (1e-8 + np.sqrt(v.dot(v)))
        result = [np.float(t) for t in list(v)]
        info_msg = 'success'
        err_msg = ''
    except Exception as e:
        #app.logger.error("error:",str(e))
        err_msg = e
        info_msg = 'error'
    response = {'msg':info_msg,'errMsg':err_msg,'result':result}
    #app.logger.error('GEN_ERROR\t' + json.dumps(response,ensure_ascii=False,indent=4))
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

def demo():
    output = {'lastTokenDense':output_layer0,'lastToken':sequence_output[:, 0, :],'sequence_vector':sequence_output}
    T = []
    inputStr = '新年快乐'
    with open(path_data,'r') as f:
        D = json.load(f)
    for i in range(491875,len(D)):
        inputStr = D[i]['content']
        text_a = inputStr
        example = InputExample(guid='guid', text_a=text_a, label='0')
        feature = convert_single_example(10, example, label_list, max_seq_length, tokenizer)
        feed_dict = {input_ids: [feature.input_ids], segment_ids: [feature.segment_ids],
                     input_mask: [feature.input_mask]}
        v0 = sess.run(sequence_output, feed_dict=feed_dict)[0]
        v = []
        for j in range(min(len(inputStr), max_seq_length - 1)):
            if inputStr[j] not in IDF:
                w = IDF['UNK']
            else:
                w = IDF[inputStr[j]]
            v.append(w * v0[j + 1])
        v = np.sum(np.array(v),axis=0)
        v = v/(1e-8+np.sqrt(v.dot(v)))
        D[i]['bert_sent2vec'] = [np.float(t) for t in list(v)]
        if i%100==0:
            print(i,len(D))

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    server = WSGIServer(("0.0.0.0", port), app)
    print("Server started")
    server.serve_forever()


#