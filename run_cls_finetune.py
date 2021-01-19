# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization_finetuning as optimization
import tokenization
import tensorflow as tf
import random
import numpy as np
import json
import time
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# from loss import bi_tempered_logistic_loss

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", '/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "model/roberta_zh_l12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", "model/roberta_zh_l12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "model/model_label/",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "model/roberta_zh_l12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 48,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class LCQMCPairClassificationProcessor(DataProcessor): # TODO NEED CHANGE2
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[2])
          text_a = tokenization.convert_to_unicode(line[0])
          text_b = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

class SentClassificationProcessor(DataProcessor): # TODO NEED CHANGE2
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[1])
          text_a = tokenization.convert_to_unicode(line[0])
          examples.append(
              InputExample(guid=guid, text_a=text_a, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  #tokens_a = tokenizer.tokenize(example.text_a)
  tokens_a = example.text_a
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = 0
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def focal_loss(logits, labels, alpha, epsilon=1.e-7,
               gamma=2.0,
               multi_dim=False):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]  not one-hot !!!
    :return: -alpha*(1-y)^r * log(y)
    它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
    logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

    怎么把alpha的权重加上去？
    通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

    是否需要对logits转换后的概率值进行限制？
    需要的，避免极端情况的影响

    针对输入是 (N，P，C )和  (N，P)怎么处理？
    先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

    bug:
    ValueError: Cannot convert an unknown Dimension to a Tensor: ?
    因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

    '''
    if multi_dim:
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        labels = tf.reshape(labels, [-1])
    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    # labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,use_focal=True,D_alpha={}):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  sequence_output = model.get_sequence_output()
  hidden_size = output_layer.shape[-1].value
  Loss = 0
  Probabilities = []
  for i in range(len(num_labels)):
      output_weights = tf.get_variable(
          "output_weights"+str(i), [num_labels[i], hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable(
          "output_bias"+str(i), [num_labels[i]], initializer=tf.zeros_initializer())
      with tf.variable_scope("loss"):
        if is_training:
          # I.e., 0.1 dropout
          output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels[i], depth=num_labels[i], dtype=tf.float32)
        if not use_focal:
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) # todo 08-29 try temp-loss
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss = focal_loss(logits, labels[i], D_alpha)
        ###############bi_tempered_logistic_loss############################################################################
        # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
        # t1=0.9 #t1=0.90
        # t2=1.05 #t2=1.05
        # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
        #tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
        ##############bi_tempered_logistic_loss#############################################################################

        Loss+=loss
        Probabilities.append(probabilities)
  return (sequence_output,output_layer,Loss, Probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,use_focal=True,D_alpha={}):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (sequence_output,output_layer,total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings,use_focal,D_alpha)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

class LCQMCPairClassificationProcessor(DataProcessor): # TODO NEED CHANGE2
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "dev") # todo change temp for test purpose

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]
    #return ["-1","0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[2])
          text_a = tokenization.convert_to_unicode(line[0])
          text_b = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

class SentencePairClassificationProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train_0827.tsv")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_0827.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_0827.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]
    #return ["-1","0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[0])
          text_a = tokenization.convert_to_unicode(line[1])
          text_b = tokenization.convert_to_unicode(line[2])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  import re
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
  init_vars = tf.train.list_variables(init_checkpoint)
  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name[3:] not in name_to_variable:
      continue
    assignment_map[name] = name[3:]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
  return (assignment_map, initialized_variable_names)
class Tokenizer(object):
    def __init__(self, vocab,stopwords=[]):
        self.vocab = self.load_vocab(vocab)
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.stopwords = stopwords
    def tokenize(self,tokens):
        res = []
        for t in tokens:
            if t in self.vocab:
                t1 = self.vocab[t]
            else:
                t1 = self.vocab['[UNK]']
            res.append(t1)
        return res
    def convert_tokens_to_ids(self, sentence, seq_length=None, begin_str = '',end_str=''):
        tokens = sentence
        res = []
        if begin_str:
            res.append(self.vocab[begin_str])
        r = []
        for t in tokens:
            if t in self.vocab:
                r.append(self.vocab[t])
            else:
                r.append(self.vocab['[UNK]'])
        res.extend(r)
        if seq_length:
            if seq_length<len(res):
                res = res[:seq_length]
            else:
                res += (seq_length - len(res)) * [self.vocab['[PAD]']]
            if end_str:
                res[-1] = self.vocab[end_str]
        else:
            if end_str:
                res.append(self.vocab[end_str])
        return res
    def convert_sentences_to_ids(self,sentences, seq_length=None, begin_str = '[B]',end_str='[E]', split_str = '[SEP]'):
        res = []
        for i in range(len(sentences)):
            if i!=len(sentences)-1:
                es = split_str
            else:
                es = end_str
            if i==0:
                bs = begin_str
            else:
                bs = ''
            s0 = self.convert_tokens_to_ids(sentences[i], seq_length=None, begin_str = bs,end_str=es)
            res.extend(s0)
        if seq_length:
            if seq_length < len(res):
                res = res[:seq_length]
            else:
                res += (seq_length - len(res)) * [self.vocab['[PAD]']]
            res[-1] = self.vocab[end_str]
        else:
            if end_str:
                res.append(self.vocab[end_str])
        return res
    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[temp] for temp in ids]
    def load_vocab(self,vocab_file):
        if isinstance(vocab_file,list):
            vocab = {vocab_file[i]:i for i in range(len(vocab_file))}
            if '[UNK]' not in vocab:
                vocab['[UNK]'] = len(vocab)
            if '[PAD]' not in vocab:
                vocab['[PAD]'] = len(vocab)
            return vocab
        with open(vocab_file,'r',encoding='utf-8') as f:
            V = f.read().strip().split('\n')
        vocab = {}
        for v in V:
            vocab[v] = len(vocab)
        if '[PAD]' not in vocab:
            vocab['[PAD]'] = len(vocab)
        if '[UNK]' not in vocab:
            vocab['[UNK]'] = len(vocab)
        if '[CLS]' not in vocab:
            vocab['[CLS]'] = len(vocab)
        if '[SEP]' not in vocab:
            vocab['[SEP]'] = len(vocab)
        if '[MASK]' not in vocab:
            vocab['[MASK]'] = len(vocab)
        if '[B]' not in vocab:
            vocab['[B]'] = len(vocab)
        if '[E]' not in vocab:
            vocab['[E]'] = len(vocab)
        return vocab

def get_model(max_seq_length, L, D_map, batch_size=64, is_training=True,use_focal=True,D_alpha={}):
    # seq_len = 32
    # vocab_size = 10000
    # dim_emb = 128
    learning_rate = 1e-4
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.reset_default_graph()
    input = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name='segment_ids')
    Y = []
    for i in range(len(L)):
        y = tf.placeholder(tf.int64, [batch_size, ], name="input-y-" + str(i))
        Y.append(y)
    num_labels = [len(D_map[k]) for k in L]
    sequence_output,output_layer,Loss, Predict= create_model(bert_config, is_training, input, input_mask, segment_ids,
                 Y, num_labels, use_one_hot_embeddings=False,use_focal=use_focal,D_alpha=D_alpha)
    Acc = 0
    for i in range(len(L)):
        correct_prediction = tf.equal(tf.argmax(Predict[i], 1), Y[i])
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        Acc += accuracy
    Acc = Acc / len(L)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(Loss)
    return input,input_mask,segment_ids, Y, Loss, Acc, train_op, Predict
def iter_data(path_data,tokenizer,seq_len,L0,idx0,batch_size=64,epochs = 10, mode = 'train'):
    X_input_ids = []
    X_segment_ids = []
    X_input_mask = []
    Y = [[] for _ in range(len(L0))]
    with open(path_data,'r',encoding='utf-8') as f:
        S = f.read().strip().split('\n')
    S = [s.split('\t') for s in S]
    epoch = 0
    while True:
        random.shuffle(S)
        if mode == 'train' and epoch>=epochs:
            break
        for s in S:
            example = InputExample(guid='guid', text_a=s[0], label='0')
            feature = convert_single_example(10, example, seq_len, tokenizer)
            X_input_ids.append(feature.input_ids)
            X_segment_ids.append(feature.segment_ids)
            X_input_mask.append(feature.input_mask)
            for i in range(len(L0)):
                Y[i].append(int(s[idx0[i]+1]))
            if len(X_input_ids)>=batch_size:
                yield X_input_ids,X_segment_ids,X_input_mask,Y,epoch
                X_input_ids = []
                X_segment_ids = []
                X_input_mask = []
                Y = [[] for _ in range(len(L0))]
        epoch+=1
    yield "__STOP__"
def getdata(path_data,tokenizer,seq_len,L0,idx0):
    X_input_ids = []
    X_segment_ids = []
    X_input_mask = []
    Y = [[] for _ in range(len(L0))]
    with open(path_data,'r',encoding='utf-8') as f:
        S = f.read().strip().split('\n')
    S = [s.split('\t') for s in S]
    for s in S:
        example = InputExample(guid='guid', text_a=s[0], label='0')
        feature = convert_single_example(10, example, seq_len, tokenizer)
        X_input_ids.append(feature.input_ids)
        X_segment_ids.append(feature.segment_ids)
        X_input_mask.append(feature.input_mask)
        for i in range(len(L0)):
            Y[i].append(int(s[idx0[i]+1]))
    return X_input_ids,X_segment_ids,X_input_mask,Y
def model_eval(session,data_dev,input,segment_ids,input_mask,Y,Loss,Acc,Predict,L,batch_size):
    Loss0 = []
    Acc_dev = []
    Yp = [[]]*len(L)
    X_input_ids_,X_segment_ids_,X_input_mask_,Y_d = data_dev
    feed_dict = {}
    i = 0
    while (i+1)*batch_size<len(X_input_ids_):
        feed_dict = {input: X_input_ids_[i*batch_size:(i+1)*batch_size], segment_ids: X_segment_ids_[i*batch_size:(i+1)*batch_size],
                     input_mask: X_input_mask_[i*batch_size:(i+1)*batch_size]}
        for j in range(len(L)):
            feed_dict[Y[j]] = Y_d[j][i*batch_size:(i+1)*batch_size]
        loss_dev, acc_dev, y_d_p = session.run([Loss, Acc, Predict], feed_dict=feed_dict)
        Loss0.append(loss_dev)
        Acc_dev.append(acc_dev)
        for j in range(len(L)):
            if i==0:
                Yp[j] = y_d_p[j]
            else:
                Yp[j] = np.concatenate((Yp[j],y_d_p[j]),axis=0)
        i+=1
    Yd = [Y_d[i][:len(Yp[i])] for i in range(len(Y_d))]
    return np.mean(Loss0),np.mean(Acc_dev),Yd,Yp
def get_time_dif(start_time):
    from datetime import timedelta
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
def main():
    path_data = FLAGS.data_dir
    max_seq_length = FLAGS.max_seq_length
    init_checkpoint = FLAGS.init_checkpoint
    path_map = os.path.join(path_data, 'map_index.json')
    #path_vocab = os.path.join(path_data, 'vocab.txt')
    path_vocab = FLAGS.vocab_file
    path_alpha = os.path.join(path_data, 'label_alpha.json')
    path_model = FLAGS.output_dir
    print_steps = 100
    save_steps = 1000
    is_training = FLAGS.do_train
    train_batch_size = FLAGS.train_batch_size
    L0 = ['使用场景P0', '表达对象P0', '表达者性别倾向P0', '文字风格']
    L = FLAGS.task_name
    L = L.split(',')
    idx0 = []
    for l in L:
        idx0.append(L0.index(l))
    path_train = os.path.join(path_data, 'train.txt')
    path_dev = os.path.join(path_data, 'dev.txt')
    D_map = json.load(open(path_map, 'r'))
    D_alpha0 = json.load(open(path_alpha, 'r'))
    D_alpha = {k: [D_alpha0[k][kk] for kk in D_map[k]] for k in D_map}
    tokenizer = Tokenizer(path_vocab)
    input,input_mask,segment_ids, Y, Loss, Acc, train_op, Predict = get_model(max_seq_length, L, D_map, batch_size=None, is_training=is_training,use_focal=True,D_alpha=D_alpha[L[0]])
    saver = tf.train.Saver(max_to_keep=None)
    session = tf.Session()
    global_step = tf.train.get_or_create_global_step()
    model_train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    session.run(tf.global_variables_initializer())
    if 'bert' in init_checkpoint:
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
    else:
        saver.restore(session, init_checkpoint)
    iter = iter_data(path_train, tokenizer, max_seq_length, L, idx0, epochs=100,batch_size=train_batch_size)
    data_dev = getdata(path_dev, tokenizer, max_seq_length, L, idx0)
    data = next(iter)
    total_batch = 0  # 总批次
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    while data != '__STOP__':
        X_input_ids_,X_segment_ids_,X_input_mask_,Ybatch, epoch = data
        start_time = time.time()
        feed_dict = {input: X_input_ids_, segment_ids: X_segment_ids_,
                     input_mask: X_input_mask_}
        for i in range(len(L)):
            feed_dict[Y[i]] = Ybatch[i]
        loss_train, acc_train, _ = session.run([Loss, Acc, model_train_op], feed_dict=feed_dict)
        if total_batch % print_steps == 0:
            # 每多少轮次输出在训练集和验证集上的性能
            time_dif = get_time_dif(start_time)
            msg = 'EPOCH:{},total_batch:{},loss_train:{},acc_train:{},time_diff:{}'
            print(msg.format(epoch, total_batch, '%0.2f' % loss_train, '%0.2f' % acc_train, time_dif))
        if total_batch % save_steps == 0:
            loss_dev, acc_dev, Y_d, Y_p = model_eval(session, data_dev, input,segment_ids,input_mask, Y, Loss, Acc, Predict, L,
                                                     train_batch_size)
            for j in range(len(Y_d)):
                y_one_hot = label_binarize(Y_d[j], np.arange(len(D_map[L[j]])))
                yy = np.array(Y_p[j])
                yy = np.argmax(yy, axis=-1)
                acc = sum([yy[k] == Y_d[j][k] for k in range(len(yy))]) / len(yy)
                fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), Y_p[j].ravel())
                auc = metrics.auc(fpr, tpr)
                print('dev %s acc : %0.2f, auc:%0.2f' % (L[j], acc, auc))
            # auc = calAUC(y_d_p, y_d)
            # if auc>auc_dev0:
            #     auc_dev0=auc
            #     global_step = self.session.run(self.global_step)
            #     print('AUC update with auc=%0.2f in global_step %d'%(auc,global_step))
            # if acc_dev > acc_dev0:
            # self.saver.save(self.session, os.path.join(self.path_model, 'model.ckpt'), global_step=self.global_step)
            saver.save(session, os.path.join(path_model, 'model.ckpt'),
                       global_step=global_step)
            msg = 'EPOCH:{},total_batch:{},loss_dev:{},acc_dev:{},time_diff:{}'
            time_dif = get_time_dif(start_time)
            print(msg.format(epoch, total_batch, '%0.2f' % loss_dev, '%0.2f' % acc_dev, time_dif))
        total_batch += 1
        data = next(iter)
if __name__ == "__main__":
    main()