from bertSentEmb import *
import numpy as np
from sklearn import preprocessing
import json
import sys
import os
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
bert_config_file,vocab_file,init_checkpoint,max_seq_length = sys.argv[2:]
max_seq_length = int(max_seq_length)
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
def sentEmb(S):
    text_a = S
    example = InputExample(guid='guid', text_a=text_a, label='0')
    feature = convert_single_example(10, example, label_list, max_seq_length, tokenizer)
    feed_dict = {input_ids: [feature.input_ids], segment_ids: [feature.segment_ids],
                 input_mask: [feature.input_mask]}
    v0 = sess.run(sequence_output, feed_dict=feed_dict)
    v = []
    for j in range(min(len(S), max_seqlen - 1)):
        if S[j] not in IDF:
            w = IDF['UNK']
        else:
            w = IDF[S[j]]
        v.append(w * v0[j + 1])
    y = np.sum(np.array(v), axis=0)
    return y
def sentEmbing():
    pass
if __name__=='__main__':
    path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag, path_idf = sys.argv[2:]
    max_seqlen = int(max_seqlen)
    IDF = json.load(open(path_idf))
    main(path_data, path_target, init_checkpoint, bert_config_file, vocab_file, max_seqlen, tag, IDF)
