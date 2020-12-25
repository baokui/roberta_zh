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
import math
import numpy as np
# from loss import bi_tempered_logistic_loss

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "data_allScene_pretrain/raw-washed",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "model/model_s2v/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", "model/model_s2v/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "model/model_s2v/ckpt3",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "model/model_s2v/ckpt_init/model.ckpt",
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

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
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

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
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

  label_id = label_map[example.label]
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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
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

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) # todo 08-29 try temp-loss
    ###############bi_tempered_logistic_loss############################################################################
    # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
    # t1=0.9 #t1=0.90
    # t2=1.05 #t2=1.05
    # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
    #tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
    ##############bi_tempered_logistic_loss#############################################################################

    loss = tf.reduce_mean(per_example_loss)

    return (sequence_output,output_layer,loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
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
        num_labels, use_one_hot_embeddings)

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
def iterData(path_file,tokenizer,batch_size=64,epochs=3):
    import random
    label_list = ['0','1']
    files = os.listdir(path_file)
    files = [os.path.join(path_file,file) for file in files]
    X_input = []
    X_mask = []
    X_segmet = []
    word_inputs = []
    word_labels = []
    for epoch in range(epochs):
        random.shuffle(files)
        for i in range(len(files)):
            with open(files[i],'r',encoding='utf-8') as f:
                for line in f:
                    text_a = line.strip()
                    example = InputExample(guid='guid', text_a=text_a, label='0')
                    feature = convert_single_example(10, example, label_list, FLAGS.max_seq_length, tokenizer)
                    x = feature.input_ids
                    for j in range(1,len(x)):
                        t = x[j]
                        if t==0:
                            break
                        if random.random()>0.2:
                            continue
                        word_inputs.append(t)
                        word_labels.append(x[j+1])
                        X_input.append(feature.input_ids)
                        X_segmet.append(feature.segment_ids)
                        X_mask.append(feature.input_mask)
                        if len(X_input)>=batch_size:
                            yield epoch,X_input,X_mask,X_segmet,word_inputs,word_labels
                            X_input = []
                            X_mask = []
                            X_segmet = []
                            word_inputs = []
                            word_labels = []
    yield '__STOP__'
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
def main(_):
    embedding_size = 768
    num_sampled = 128
    learning_rate = 1e-3
    step_printloss = 100
    step_savemodel = 1000
    dim_cls = 256
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    vocabulary_size = len(tokenizer.vocab)
    tf.reset_default_graph()
    input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    word_inputs = tf.placeholder(tf.int32, shape=[None])
    word_labels = tf.placeholder(tf.int32, shape=[None, 1])
    model = modeling.BertModel(
        config=bert_config,
        is_training=FLAGS.do_train,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    sequence_output = model.get_sequence_output()
    #first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    first_token_tensor = tf.reduce_mean(sequence_output, axis=1)
    out1 = tf.layers.dense(first_token_tensor, units=dim_cls, activation=tf.nn.relu,name='fine_s2v_cls')
    embeddings = model.get_embedding_table()
    embed = tf.nn.embedding_lookup(embeddings, word_inputs)
    feature = tf.concat([out1, embed], axis=-1)
    #a = 0.01
    #feature = a*embed+(1-a)*first_token_tensor
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, dim_cls+embedding_size],
                            stddev=1.0 / math.sqrt(dim_cls+embedding_size)),name='fine_s2v_cce_w')
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32,name='fine_s2v_cce_b')
    nce_weights0 = nce_weights[:,:dim_cls]
    nce_weights1 = nce_weights[:,dim_cls:]
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=feature, labels=word_labels,
                       num_sampled=num_sampled, num_classes=vocabulary_size))
    logits = tf.matmul(feature,nce_weights,transpose_b=True)+nce_biases
    pro_predict = tf.nn.softmax(logits,axis=-1)
    label_predict = tf.argmax(pro_predict,axis=-1)

    logits_emb = tf.matmul(embed, nce_weights1, transpose_b=True) + nce_biases
    pro_predict_emb = tf.nn.softmax(logits_emb, axis=-1)
    label_predict_emb = tf.argmax(pro_predict_emb, axis=-1)
    logits_cls = tf.matmul(out1, nce_weights0, transpose_b=True) + nce_biases
    pro_predict_cls = tf.nn.softmax(logits_cls, axis=-1)
    label_predict_cls = tf.argmax(pro_predict_cls, axis=-1)

    update_var_list = []  # 该list中的变量参与参数更新
    tvars = tf.trainable_variables()
    for tvar in tvars:
        if "bert" not in tvar.name:
            update_var_list.append(tvar)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=update_var_list)
    saver = tf.train.Saver(max_to_keep=10)
    global_step = tf.train.get_or_create_global_step()
    model_train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if FLAGS.init_checkpoint:
        try:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
        except:
            saver.restore(sess, FLAGS.init_checkpoint)

    iter = iterData(FLAGS.data_dir, tokenizer, batch_size=FLAGS.train_batch_size,epochs=int(FLAGS.num_train_epochs))
    data = next(iter)
    while data!='__STOP__':
        epoch,batch_input_ids,batch_input_mask,batch_segment_ids,batch_word_inputs,batch_word_labels = data
        batch_word_labels = np.array(batch_word_labels)
        batch_word_labels = np.reshape(batch_word_labels,[len(batch_word_labels),1])
        feed_dict = {input_ids:batch_input_ids,input_mask:batch_input_mask,segment_ids:batch_segment_ids,word_inputs:batch_word_inputs,word_labels:batch_word_labels}
        batch_loss,step,_ = sess.run([loss,global_step,model_train_op],feed_dict=feed_dict)
        if step%step_printloss==0:
            print('train total steps %d (epoch %d) with loss=%0.4f'%(step,epoch,batch_loss))
        if step % step_savemodel == 0:
            saver.save(sess, os.path.join(FLAGS.output_dir, 'model.ckpt'),
                       global_step=global_step)
            # import random
            # random.shuffle(batch_input_ids)
            # feed_dict = {input_ids: batch_input_ids, input_mask: batch_input_mask, segment_ids: batch_segment_ids,
            #              word_inputs: batch_word_inputs, word_labels: batch_word_labels}
            label_predict_,label_predict_emb_,label_predict_cls_ = sess.run([label_predict,label_predict_emb,label_predict_cls],feed_dict=feed_dict)
            acc = sum([label_predict_[j]==batch_word_labels[j][0] for j in range(len(label_predict_))])/(0.01+len(label_predict_))
            acc_emb = sum([label_predict_emb_[j] == batch_word_labels[j][0] for j in range(len(label_predict_))]) / (
                        0.01 + len(label_predict_))
            acc_cls = sum([label_predict_cls_[j] == batch_word_labels[j][0] for j in range(len(label_predict_))]) / (
                    0.01 + len(label_predict_))
            print('acc:%0.4f,%0.4f,%0.4f'%(acc,acc_emb,acc_cls))
            #embed_,first_token_tensor_,feature_ = sess.run([embed,first_token_tensor,feature],feed_dict = feed_dict)
        data = next(iter)

def sentEmb(S):
    embedding_size = 768
    dim_cls = 256
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    vocabulary_size = len(tokenizer.vocab)
    tf.reset_default_graph()
    input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    word_inputs = tf.placeholder(tf.int32, shape=[None])
    word_labels = tf.placeholder(tf.int32, shape=[None, 1])
    model = modeling.BertModel(
        config=bert_config,
        is_training=FLAGS.do_train,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    sequence_output = model.get_sequence_output()
    #first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    first_token_tensor = tf.reduce_mean(sequence_output, axis=1)
    out1 = tf.layers.dense(first_token_tensor, units=dim_cls, activation=tf.nn.relu, name='fine_s2v_cls')
    embeddings = model.get_embedding_table()
    embed = tf.nn.embedding_lookup(embeddings, word_inputs)
    feature = tf.concat([out1, embed], axis=-1)
    # a = 0.01
    # feature = a*embed+(1-a)*first_token_tensor
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, dim_cls + embedding_size],
                            stddev=1.0 / math.sqrt(dim_cls + embedding_size)), name='fine_s2v_cce_w')
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32, name='fine_s2v_cce_b')
    nce_weights0 = nce_weights[:, :dim_cls]
    nce_weights1 = nce_weights[:, dim_cls:]
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=feature, labels=word_labels,
                       num_sampled=num_sampled, num_classes=vocabulary_size))
    logits = tf.matmul(feature, nce_weights, transpose_b=True) + nce_biases
    pro_predict = tf.nn.softmax(logits, axis=-1)
    label_predict = tf.argmax(pro_predict, axis=-1)

    logits_emb = tf.matmul(embed, nce_weights1, transpose_b=True) + nce_biases
    pro_predict_emb = tf.nn.softmax(logits_emb, axis=-1)
    label_predict_emb = tf.argmax(pro_predict_emb, axis=-1)
    logits_cls = tf.matmul(out1, nce_weights0, transpose_b=True) + nce_biases
    pro_predict_cls = tf.nn.softmax(logits_cls, axis=-1)
    label_predict_cls = tf.argmax(pro_predict_cls, axis=-1)
    output = {'sent2vec':out1}
    saver = tf.train.Saver(max_to_keep=None)
    sess = tf.Session()
    module_file = tf.train.latest_checkpoint(FLAGS.output_dir)
    saver.restore(sess, module_file)
    # X_input = []
    # X_mask = []
    # X_segmet = []
    # for i in range(len(S)):
    #     text_a = S[i]
    #     example = InputExample(guid='guid', text_a=text_a, label='0')
    #     feature = convert_single_example(10, example, ['0','1'], FLAGS.max_seq_length, tokenizer)
    #     X_input.append(feature.input_ids)
    #     X_segmet.append(feature.segment_ids)
    #     X_mask.append(feature.input_mask)
    T = []
    for i in range(len(S)):
        if i % 100 == 0:
            print(i, len(S))
        text_a = S[i]
        example = InputExample(guid='guid', text_a=text_a, label='0')
        feature = convert_single_example(10, example, ['0','1'], FLAGS.max_seq_length, tokenizer)
        feed_dict = {input_ids: [feature.input_ids], segment_ids: [feature.segment_ids],
                     input_mask: [feature.input_mask]}
        y = {key: sess.run(output[key], feed_dict=feed_dict)[0] for key in output}
        T.append([i, S[i], y])
    return T

if __name__ == "__main__":
  # flags.mark_flag_as_required("data_dir")
  # flags.mark_flag_as_required("task_name")
  # flags.mark_flag_as_required("vocab_file")
  # flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  tf.app.run()