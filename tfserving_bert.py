from bertSentEmb import modeling,create_model,InputExample,convert_single_example
import tokenization
import tensorflow as tf
import os
import requests
import json
import sys
import time
import numpy as np
from sklearn import preprocessing
def norm(V1):
    V1 = preprocessing.scale(V1, axis=-1)
    V1 = V1 / np.sqrt(len(V1[0]))
    return V1
def pbmodel(bert_config_file,checkpoint_path,max_seq_length,path_export_model,version='0'):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tf.reset_default_graph()
    input_ids = tf.placeholder(tf.int32,shape = [1,max_seq_length],name = 'input_ids')
    #input_mask = tf.placeholder(tf.int32,shape = [None,max_seq_length],name = 'input_mask')
    #segment_ids = tf.placeholder(tf.int32,shape = [None,max_seq_length],name = 'segment_ids')
    input_mask = tf.zeros([1,max_seq_length],tf.int32,name = 'input_mask')
    segment_ids = tf.zeros([1,max_seq_length],tf.int32,name = 'segment_ids')
    labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
    sequence_output,output_layer0,loss, per_example_loss, logits, probabilities = create_model(bert_config, False, input_ids, input_mask, segment_ids,labels, 2, False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    print("init_checkpoint:", checkpoint_path)
    if checkpoint_path:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_path)
        tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path_export_model,version))
    signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'feat_index': input_ids},
                                                                         outputs={'scores': sequence_output})
    builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={'predict': signature,
                                                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print(path_export_model + "/" + version)
    return input_ids,input_mask,segment_ids,sequence_output
def test(path_data,path_target,vocab_file,max_seq_length=48):
    url = 'http://localhost:8503/v1/models/bert3:predict'
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    with open(path_data,'r',encoding='utf-8') as f:
        D = json.load(f)
    t0 = time.time()
    for i in range(len(D)):
        if i % 100 == 0:
            print(i, len(D),time.time()-t0)
        text_a = D[i]['content']
        example = InputExample(guid='guid', text_a=text_a, label='0')
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
        assert len(input_ids) == max_seq_length
        r = requests.post(url=url,data=json.dumps({'instances':[input_ids]}))
        p = r.json()['predictions'][0]
        y = p[1:len(text_a)+1]
        y = np.mean(np.array(y),axis=0)
        y = norm(y)
        D[i]['sent_bert3'] = list(y)
        if i%10000==0:
            with open(path_target,'w',encoding='utf-8') as f:
                json.dump(D,f,ensure_ascii=False,indent=4)
    with open(path_target, 'w', encoding='utf-8') as f:
        json.dump(D, f, ensure_ascii=False, indent=4)
def save():
    bert_config_file = 'model/bert_prose_finetune_mgpu/bert_config.json'
    checkpoint_path = 'model/bert_prose_finetune_mgpu/ckpt/model.ckpt'
    max_seq_length = 48
    path_export_model = 'model/bert_prose_finetune_mgpu/pbmodel/'
    version = '0'
    pbmodel(bert_config_file, checkpoint_path, max_seq_length, path_export_model, version)
if __name__=='__main__':
    if sys.argv[-1] == 'test':
        path_data,path_target = sys.argv[1:3]
        vocab_file = 'model/bert_prose_finetune_mgpu/vocab.txt'
        test(path_data,path_target,vocab_file)